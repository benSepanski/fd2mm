from warnings import warn  # noqa
from abc import abstractmethod
from collections import defaultdict
import numpy as np

from meshmode.mesh import NodalAdjacency, BTAG_ALL, BTAG_REALLY_ALL
from fd2mm.analogs import Analog
from fd2mm.analogs.finat_element import FinatElementAnalog
from fd2mm.analogs.functionspaceimpl import FunctionSpaceAnalog
from fd2mm.analogs.function import CoordinatelessFunctionAnalog
from fd2mm.analogs.utils import compute_orientation, reorder_nodes, cached
from firedrake.mesh import MeshTopology


class MeshTopologyAnalog(Analog):

    def __init__(self, mesh):
        """
            :arg mesh: A :mod:`firedrake` :class:`MeshTopology` or
                       :class:`MeshGeometry`.

            We require that :arg:`mesh` have co-dimesnion
            of 0 or 1.
            Moreover, if :arg:`mesh` is a 2-surface embedded in 3-space,
            we _require_ that :function:`init_cell_orientations`
            has been called already.
        """
        top = mesh.topological

        # {{{ Check input
        if not isinstance(mesh, MeshTopology):
            raise TypeError(":arg:`mesh` must be of type "
                            ":class:`firedrake.mesh.MeshTopology` or "
                            ":class:`firedrake.mesh.MeshGeometry`")
        # }}}

        super(MeshTopologyAnalog, self).__init__(top)

        # Ensure has simplex-type elements
        if not mesh.ufl_cell().is_simplex():
            raise ValueError("mesh must have simplex type elements, "
                             "%s is not a simplex" % (mesh.ufl_cell()))

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        if self.cell_dimension() not in supported_dims:
            raise ValueError("Cell dimension is %s. Cell dimension must be one of"
                             "range %s" % (self.cell_dimension(), supported_dims))

    @property
    def topology_a(self):
        return self

    @property
    def topological_a(self):
        return self

    def cell_dimension(self):
        """
            Return the dimension of the cells used by this topology
        """
        return self.analog().cell_dimension()

    def nelements(self):
        return self.analog().num_cells()

    def nunit_vertices(self):
        return self.analog().ufl_cell().num_vertices()

    def bdy_tags(self):
        """
            Return a tuple of bdy tags as requested in
            the construction of a :mod:`meshmode` :class:`Mesh`

            The tags used are :class:`meshmode.mesh.BTAG_ALL`,
            :class:`meshmode.mesh.BTAG_REALLY_ALL`, and
            any markers in the mesh topology's exterior facets
            (see :attr:`firedrake.mesh.MeshTopology.exterior_facets.unique_markers`)
        """
        bdy_tags = [BTAG_ALL, BTAG_REALLY_ALL]

        unique_markers = self.analog().exterior_facets.unique_markers
        if unique_markers is not None:
            bdy_tags += unique_markers

        return tuple(bdy_tags)

    def nodal_adjacency(self):
        plex = self.analog()._plex
        cStart, cEnd = plex.getHeightStratum(0)
        vStart, vEnd = plex.getDepthStratum(0)

        element_to_neighbors = {}

        # For each vertex
        for vert_id in range(vStart, vEnd):
            # Record all cells touching vertex
            cells = []
            for cell_id in plex.getTransitiveClosure(vert_id, useCone=False)[0]:
                if cStart <= cell_id < cEnd:
                    firedrake_id = self.analog()._cell_numbering.getOffset(cell_id)
                    cells.append(firedrake_id)

            for cell_one in cells:
                element_to_neighbors.setdefault(cell_one, [])
                for cell_two in cells:
                    element_to_neighbors[cell_one].append(cell_two)

        # Create neighbor_starts and neighbors
        neighbors = []
        neighbor_starts = np.zeros(self.nelements() + 1)
        for iel in range(len(element_to_neighbors)):
            elt_neighbors = element_to_neighbors[iel]
            neighbors.append(elt_neighbors)
            neighbor_starts[iel+1] = len(neighbors)

        return NodalAdjacency(neighbor_starts=neighbor_starts,
                              neighbors=neighbors)


class MeshGeometryAnalog(Analog):
    """
        This takes a :mod:`firedrake` :class:`MeshGeometry`
        and converts its data so that :mod:`meshmode` can handle it.

        NOTE: This an analog of a :mod:`firedrake` mesh, but NOT
              an analog of a :mod:`meshmode` mesh. In particular,
              it doesn't store node information, just vertex
              information. If you are wanting a :mod:`meshmode` mesh,
              you really need a :class:`FunctionSpaceAnalog`.
    """

    def __init__(self, mesh, coordinates_analog):
        """
            :arg mesh: A :mod:`firedrake` :class:`MeshGeometry`.
                We require that :arg:`mesh` have co-dimesnion
                of 0 or 1.
                Moreover, if :arg:`mesh` is a 2-surface embedded in 3-space,
                we _require_ that :function:`init_cell_orientations`
                has been called already.

            :arg mesh_topology_analog: A :class:`MeshTopologyAnalog` to use

            :arg coordinates_analog: A :class:`CoordinatelessFunctionSpaceAnalog`
                                      to use, represents the coordinates

        """
        super(MeshGeometryAnalog, self).__init__(mesh)

        # {{{ Make sure input data is valid

        # Ensure is not a topological mesh
        if mesh.topological == mesh:
            raise TypeError(":arg:`mesh` must be of type"
                            " :class:`firedrake.mesh.MeshGeometry`")

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        if mesh.geometric_dimension() not in supported_dims:
            raise ValueError("Geometric dimension is %s. Geometric "
                             " dimension must be one of range %s"
                             % (mesh.geometric_dimension(), supported_dims))

        # Raise warning if co-dimension is not 0 or 1
        co_dimension = mesh.geometric_dimension() - mesh.topological_dimension()
        if co_dimension not in [0, 1]:
            raise ValueError("Codimension is %s, but must be 0 or 1." %
                             (co_dimension))

        # Ensure coordinates are coordinateless
        if not isinstance(coordinates_analog, CoordinatelessFunctionAnalog):
            raise ValueError(":arg:`coordinates_analog` must be of type"
                             " CoordinatelessFunctionAnalog")

        topology_a = coordinates_analog.function_space_a().mesh_a()

        if not topology_a.is_analog(mesh.topology):
            raise ValueError("Topology coordinates live on must be same "
                             "topology as :arg:`mesh` lives on")

        # }}}

        # For sharing data like in firedrake
        self._shared_data_cache = defaultdict(dict)

        # Store input information
        self._coordinates_a = coordinates_analog
        self._topology_a = topology_a

        # Create attributes to be computed later
        self._vertex_indices = None           # See method vertex_indices()
        self._vertices = None                 # See method vertices()
        self._orient = None                   # See method orientation()
        self._facial_adjacency_groups = None

        self._no_init = True

    def __getattr__(self, attr):
        """
        Done like :class:`MeshGeometry`
        """
        return getattr(self._topology_a, attr)

    @cached
    def _coordinates_function_a(self, key=None):
        assert key is None

        from fd2mm.analogs.functionspaceimpl import WithGeometryAnalog
        from fd2mm.analogs.function import FunctionAnalog
        from firedrake.function import Function

        coordinates_fs = self._coordinates_a.analog().function_space()
        coordinates_fs_a = self._coordinates_a.function_space_a()

        V_a = WithGeometryAnalog(coordinates_fs, coordinates_fs_a, self)
        f = Function(V_a.analog(), val=self._coordinates_a.analog())
        f_a = FunctionAnalog(f, V_a)
        return f_a

    def coordinates_a(self):
        """
            Return coordinates as a function
        """
        return self._coordinates_function_a()

    def coordinates_finat_element_analog(self):
        return self._coordinates_finat_element_analog

    def init(self, normals=None, no_normals_warn=True):
        """
            Compute vertex indices, vertices, and orientations

            For args see :func:`fd2mm.analogs.utils.compute_orientation`
        """
        if self._no_init:
            self._no_init = False

            # Convert cell node list of mesh to vertex list
            unit_vertex_indices = self._finat_element_analog.unit_vertex_indices()
            cell_node_list = \
                self.analog().coordinates.function_space().cell_node_list

            vertex_indices = cell_node_list[:, unit_vertex_indices]

            # Get maps new vert numbering->old and old->new
            vert_ndx_to_fd_ndx = np.unique(np.flatten(vertex_indices))
            fd_ndx_to_vert_ndx = dict(zip(vert_ndx_to_fd_ndx,
                                          np.arange(vert_ndx_to_fd_ndx.shape[0])
                                          ))
            # Get vertices array
            vertices = np.real(
                self.analog().coordinates.dat.data[vert_ndx_to_fd_ndx])
            #:mod:`meshmode` wants [ambient_dim][nvertices], but for now we
            #write it as [geometric dim][nvertices]
            vertices = vertices.T.copy()

            # Use new labels on vertex indices
            vertex_indices = np.vectorize(fd_ndx_to_vert_ndx.get)(vertex_indices)

            # compute orientations
            self._orient = compute_orientation(self.analog(), vertices,
                                               vertex_indices,
                                               normals=normals,
                                               no_normals_warn=no_normals_warn)
            #Make sure the mesh fell into one of the cases
            """
              NOTE : This should be guaranteed by previous checks,
                     but is here anyway in case of future development.
            """
            assert self._orient is not None

            # apply orientation to vertex indices
            flip_matrix = self.coordinates_finat_element_analog().flip_matrix()
            reorder_nodes(self._orient, vertex_indices, flip_matrix)

            # store vertex indices and vertices
            self._vertices = vertices
            self._vertex_indices = vertex_indices

    def vertex_indices(self):
        """
            (analog to mesh.coordinates.function_space().cell_node_list)

            <from :mod:`meshmode` docs:>
            An array of (nelements, ref_element.nvertices)
            of (mesh-wide) vertex indices
        """
        self.init()
        return self._vertex_indices

    def vertices(self):
        """
        An array holding the coordinates of the vertices of shape
        (geometric dim, nvertices)
        """
        self.init()
        return self._vertices

    def orientations(self):
        """
            Return the orientations of the mesh elements:
            an array, the *i*th element is > 0 if the *ith* element
            is positively oriented, < 0 if negatively oriented
        """
        self.init()
        return self._orient

    def face_vertex_indices_to_tags(self):
        """
            Return a :mod:`meshmode` list of :class:`FacialAdjacencyGroups`
            as used in the construction of a :mod:`meshmode` :class:`Mesh`
        """

        # fvi_to_tags maps frozenset(vertex indices) to tags
        fvi_to_tags = {}
        # maps faces to local vertex indices
        connectivity = self.coordinates_fs.finat_element.cell.\
            connectivity[(self.cell_dimension() - 1, 0)]

        exterior_facets = self.analog().exterior_facets

        for i, (icells, ifacs) in enumerate(zip(exterior_facets.facet_cell,
                                                exterior_facets.local_facet_number)):
            for icell, ifac in zip(icells, ifacs):
                # record face vertex indices to tag map
                cell_vertices = self.vertex_indices()
                facet_indices = connectivity[ifac]
                fvi = frozenset(cell_vertices[list(facet_indices)])
                fvi_to_tags.setdefault(fvi, [])
                fvi_to_tags[fvi].append(exterior_facets.markers[i])

        # }}}

        return fvi_to_tags

    def facial_adjacency_groups(self):
        """
            Return a :mod:`meshmode` list of :class:`FacialAdjacencyGroups`
            as used in the construction of a :mod:`meshmode` :class:`Mesh`
        """
        # TODO: Compute facial adjacency without making a group

        # {{{ Compute facial adjacency groups if not already done

        if self._facial_adjacency_groups is None:
            from meshmode.mesh.generation import make_group_from_vertices
            from meshmode.mesh import _compute_facial_adjacency_from_vertices

            group = make_group_from_vertices(self.vertices(),
                                             self.vertex_indices(), 1)
            groups = [group]

            """
                NOTE : This relies HEAVILY on the fact that elements are *not*
                       reordered at any time, and that the only reordering is done
                       by :func:`flip_simplex_element_group`
            """
            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                groups,
                self.bdy_tags(),
                np.int32, np.int8,
                face_vertex_indices_to_tags=self.face_vertex_indices_to_tags())

        # }}}

        return self._facial_adjacency_groups


def MeshAnalog(mesh):
    coords_fspace = mesh.coordinates.function_space()

    topology_a = MeshTopologyAnalog(mesh)
    finat_elt_a = FinatElementAnalog(coords_fspace.finat_element)

    coords_fspace_a = FunctionSpaceAnalog(coords_fspace, topology_a, finat_elt_a)
    coordinates_analog = CoordinatelessFunctionAnalog(mesh.coordinates,
                                                      coords_fspace_a)

    return MeshGeometryAnalog(mesh, coordinates_analog)


class MeshAnalogWithBdy(MeshGeometryAnalog):
    """
        Abstract

        A mesh analog which depends also on some bdy id
    """
    def __init__(self, mesh, bdy_id, normals=None, no_normals_warn=True):
        """
            :arg bdy_id: An *int* which is an exterior bdy_id marker,
                           for all boundaries use either the string
                           'on_bdy_id' or the :mod:`meshmode` class
                           :class:`mesh.BTAG_ALL`
        """
        raise NotImplementedError
        # verify the ids are valid markers, store bdy_id,
        if isinstance(bdy_id, int):
            valid_bdy_ids = mesh.exterior_facets.unique_markers
            assert bdy_id in valid_bdy_ids
        elif bdy_id not in ('on_boundary', BTAG_ALL):
            raise TypeError("bdy_id must be an int, 'on_bdy_id', or"
                            "meshmode.mesh.BTAG_ALL")

        if bdy_id == BTAG_ALL:
            self._bdy_id = 'on_boundary'
        else:
            self._bdy_id = bdy_id

        # Map vertex index to its index in firedrake, an array
        self._vert_index_to_fd_index = None
        # Inverse of the above map, as a dict
        self._fd_index_to_vert_index = None
        # :attr:`_vertex_indices`, but vertices labeled with fd vertex numbering
        # instead of new ones. Destroyed once :attr:`_vertex_indices`
        # is constructed
        self._fd_vertex_indices = None

        # An array mapping new cell id to cell id in firedrake
        self._cell_id_to_fd_cell_id = None
        # a dict inverting the above array
        self._fd_cell_id_to_cell_id = None

        super(MeshAnalogWithBdy, self).__init__(mesh, normals=normals,
                                                     no_normals_warn=no_normals_warn)

    def contains_bdy(self, bdy_id):
        """
            Return *True* iff contains the given bdy_id. :arg:`bdy_id`
            should be as in :arg:`bdy_id` in :meth:`__init__`.
        """
        if bdy_id == BTAG_ALL:
            bdy_id = 'on_boundary'

        # If :attr:`_bdy_id` is a bdy id not equal to :kwarg:`bdy_id`,
        # this is not an analog
        if isinstance(self._bdy_id, int) and self._bdy_id != bdy_id:
            return False

        # Otherwise if :attr:`_bdy_id` is the whole bdy and
        # bdy_id is not a valid bdy id, this is not an analog
        elif self._bdy_id == 'on_boundary':
            if bdy_id != 'on_boundary' and bdy_id not in \
                    self.analog().exterior_facets.unique_markers:
                return False

        return True

    def is_analog(self, obj, **kwargs):
        """
            :kwarg bdy_id: As in :meth:`__init__`, defaults to *None*

            Return whether or not this object is an analog for the
            given object and bdy_id
        """
        bdy_id = kwargs.get('bdy_id', None)
        return self.contains_bdy(bdy_id) and \
            super(MeshAnalogWithBdy, self).is_analog(obj)

    @abstractmethod
    def _compute_bdy_info(self):
        """
            Compute bdy info, namely :attr:`_vert_index_to_fd_index`,
            :attr:`_cell_id_to_fd_cell_id`, and :attr:`_fd_vertex_indices`
        """

    def cell_id_to_fd_cell_id(self):
        """
            Compute and return an array from new cell id to
            the cell it was in inside of firedrake
        """
        self._compute_bdy_info()
        return self._cell_id_to_fd_cell_id

    def vert_index_to_fd_index(self):
        """
            Compute and return an array from new vertex to their old
            firedrake labeling
        """
        self._compute_bdy_info()
        return self._vert_index_to_fd_index

    def fd_index_to_vert_index(self):
        """
            Compute and return a dictionary from firedrake index
            to new index
        """
        if self._fd_index_to_vert_index is None:
            self._fd_index_to_vert_index = dict(zip(
                self.vert_index_to_fd_index(),
                np.arange(self.vert_index_to_fd_index().shape[0], dtype=np.int32)
                ))
        return self._fd_index_to_vert_index

    def vertex_indices(self):
        if self._vertex_indices is None:
            # compute bdy info
            self._compute_bdy_info()

            # apply vertex relabeling to :attr:`_fd_vertex_indices`
            self._vertex_indices = \
                np.vectorize(self.fd_index_to_vert_index().get
                             )(self._fd_vertex_indices)

            # Free the no longer needed data
            del self._fd_vertex_indices
            self._fd_vertex_indices = None

        return self._vertex_indices

    def vertices(self):
        """
            Return (relabeled) vertices, including only those vertices
            mapped to by :meth:`vert_index_to_fd_index`
        """
        if self._vertices is None:
            self._vertices = np.real(self.analog().coordinates.dat.data)

            # Only use vertices of cells near the given bdy
            self._vertices = self._vertices[self.vert_index_to_fd_index()]

            #:mod:`meshmode` wants [ambient_dim][nvertices], but for now we
            #write it as [geometric dim][nvertices]
            self._vertices = self._vertices.T.copy()

        return self._vertices


class MeshAnalogNearBdy(MeshAnalogWithBdy):
    """
        As :class:`MeshAnalog`, but ONLY converts data near
        :attr:`near_bdy`

        A cell is considered "near" :attr:`near_bdy` if
        it has at least one vertex on the bdy
    """
    def _compute_bdy_info(self):
        # Compute if necessary (check both just to be safe)
        if self._vert_index_to_fd_index is None:

            # {{{ Get all the vertex indices of vertices on the bdy
            cfspace = self.analog().coordinates.function_space()
            verts_on_bdy = cfspace.boundary_nodes(self._bdy_id, 'topological')

            # }}}

            # FIXME : This still might be slow, having to iterate over all the
            #         cells
            # Get all vertices of cells with at least one vertex on a near bdy
            # Also record the cells
            mesh = cfspace.mesh()
            plex = mesh._plex
            # First dmplex vert number to 1+last
            vStart, vEnd = plex.getDepthStratum(0)
            # First dmplex cell number to 1+last
            cStart, cEnd = plex.getHeightStratum(0)

            # firedrake vertexes ordered by their dmplex ordering
            dm_relative_vert_order_to_fd = np.vectorize(
                mesh._vertex_numbering.getOffset)(np.arange(vStart, vEnd,
                                                            dtype=np.int32))
            # maps firedrake vertex to dmplex vertex
            fd_vert_to_dm = np.argsort(dm_relative_vert_order_to_fd) + vStart

            # Compute actual cells and verts near the given bdy
            verts_near_bdy = set()
            cells_near_bdy = set()
            for vert in fd_vert_to_dm[verts_on_bdy]:
                # transitive closure of the star operation
                # returns (indices, orientations) hence the `[0]`
                support = plex.getTransitiveClosure(vert, useCone=False)[0]
                for cell_dm_id in support:
                    # add any cells which contain this point
                    if cStart <= cell_dm_id < cEnd:  # ensure is a cell
                        cell_fd_id = mesh._cell_numbering.getOffset(cell_dm_id)

                        if cell_fd_id not in cells_near_bdy:
                            cells_near_bdy.add(cell_fd_id)

                            # Now add any vertexes in the cone of the cell
                            cell_support = plex.getTransitiveClosure(cell_dm_id)[0]
                            for vert_dm_id in cell_support:
                                # if is a vertex
                                if vStart <= vert_dm_id < vEnd:
                                    vert_fd_id = \
                                        mesh._vertex_numbering.getOffset(vert_dm_id)
                                    verts_near_bdy.add(vert_fd_id)

            # Convert to list,
            self._vert_index_to_fd_index = np.array(list(verts_near_bdy),
                                                    dtype=np.int32)
            # Convert cells near bdy to numpy array
            self._cell_id_to_fd_cell_id = np.array(list(cells_near_bdy),
                                                   dtype=np.int32)

            self._fd_vertex_indices = \
                cfspace.cell_node_list[self._cell_id_to_fd_cell_id]

    def _cell_vertices(self, icell):
        if self._fd_cell_id_to_cell_id is None:
            self._fd_cell_id_to_cell_id = dict(zip(
                self.cell_id_to_fd_cell_id(),
                np.arange(self.cell_id_to_fd_cell_id().shape[0], dtype=np.int32)
                ))

        if icell not in self._fd_cell_id_to_cell_id:
            return None

        return self.vertex_indices()[self._fd_cell_id_to_cell_id[icell]]


class MeshAnalogOnBdy(MeshAnalogWithBdy):
    """
        Analog of just the given bdy, will reduce topological
        dimension by 1.

        We require the initial topological dimension equals the
        geometric dimension
    """
    def __init__(self, mesh, bdy_id, normals=None, no_normals_warn=True):
        super(MeshAnalogOnBdy, self).__init__(mesh, bdy_id, normals=normals,
                                                   no_normals_warn=no_normals_warn)

        raise ValueError("On Boundary Unsupported")
        # make dimension check
        assert self.topological_dimension() == self.geometric_dimension()
        # Lower topological dimension
        self._tdim -= 1

    def _compute_bdy_info(self):
        if self._vert_index_to_fd_index is None:
            # {{{ Get all the vertex indices of vertices on the bdy

            cfspace = self.analog().coordinates.function_space()
            self._vert_index_to_fd_index = cfspace.boundary_nodes(self._bdy_id,
                                                                  'topological')

            # }}}

            # {{{ Compute facets (which will be the new cells) and the cells
            #     they came from

            exterior_facets = self.analog().exterior_facets
            cfspace = self.analog().coordinates.function_space()
            cell_node_list = cfspace.cell_node_list
            connectivity = cfspace.finat_element.cell.connectivity[
                (self.topological_dimension(), 0)]

            self._fd_vertex_indices = []
            self._cell_id_to_fd_cell_id = []
            for marker, icells, ifacs in zip(exterior_facets.markers,
                                             exterior_facets.facet_cell,
                                             exterior_facets.local_facet_number):
                # if facet is on the bdy
                if self.contains_bdy(marker):
                    for icell, ifac in zip(icells, ifacs):
                        # Compute facet vertexes
                        cell = cell_node_list[icell]
                        facet_local_indices = connectivity[ifac]
                        facet = cell[list(facet_local_indices)]

                        self._fd_vertex_indices.append(facet)
                        self._cell_id_to_fd_cell_id.append(icell)

            self._fd_vertex_indices = np.array(self._fd_vertex_indices,
                                               dtype=np.int32)
            self._cell_id_to_fd_cell_id = np.array(self._cell_id_to_fd_cell_id,
                                                   dtype=np.int32)

            # }}}

    def bdy_tags(self):
        # Just want meshmode defaults
        if self._bdy_tags is None:
            self._bdy_tags = (BTAG_ALL, BTAG_REALLY_ALL)
        return self._bdy_tags

    def orientations(self):
        # {{{ Compute the orientations if necessary
        if self._orient is None:

            # TODO: This is probably inefficient design... but some of the
            #       computation needs a :mod:`meshmode` group
            #       right now.
            from meshmode.mesh.generation import make_group_from_vertices

            # Make a new cell node list with [facet, vertex not on facet]
            # as the vertex indices
            coords = self.analog().coordinates

            cell_node_list = coords.function_space().cell_node_list
            cells = cell_node_list[self.cell_id_to_fd_cell_id()]

            num_verts = self.vertices().shape[1]
            new_shape = (self.vertices().shape[0],
                         num_verts + cells.shape[0])
            new_vertices = np.resize(self.vertices(), new_shape)

            for i, cell in enumerate(cells):
                verts_on_facet = \
                    self.vert_index_to_fd_index()[self.vertex_indices()[i]]
                # Get the vertex not on the facet and append to new_vertices
                for vert in cell:
                    if vert not in verts_on_facet:
                        new_vertices[:, num_verts] = coords.dat.data[vert][:].real

                # Change cell to [verts_on_facet, vert_not_on_facet]
                cells[i][:-1] = self.vertex_indices()[i]
                cells[i][-1] = num_verts
                num_verts += 1

            # Just used to get orientations
            group = make_group_from_vertices(new_vertices,
                                             cells, 1)

            # We use :mod:`meshmode` to check our orientations, because
            # we already asserted gdim equal to tdim
            from meshmode.mesh.processing import \
                find_volume_mesh_element_group_orientation

            self._orient = \
                find_volume_mesh_element_group_orientation(new_vertices, group)

            # now make the group that we'll actually use later
            self._group = make_group_from_vertices(self.vertices(),
                                                   self.vertex_indices(), 1)

        return self._orient

    def _cell_vertices(self, icell):
        # We don't want to mark any bdy tags
        return None
