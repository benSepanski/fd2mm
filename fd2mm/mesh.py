from warnings import warn  # noqa
from collections import defaultdict
import numpy as np

from fd2mm.analog import Analog
from fd2mm.finat_element import FinatElementAnalog

from firedrake.mesh import MeshTopology
from meshmode.mesh import NodalAdjacency, BTAG_ALL, BTAG_REALLY_ALL


class MeshTopologyAnalog(Analog):
    """
        An analog of a :class:`firedrake.mesh.MeshTopology`.
        Holds the topological (as opposed to geometric) information
        about a mesh
    """

    def __init__(self, mesh, cells_to_use=None):
        """
            :arg mesh: A :mod:`firedrake` :class:`MeshTopology` or
                       :class:`MeshGeometry`.
            :arg cells_to_use: Either

                * *None*, in which case this argument is ignored
                * An array of cell ids, in which case those are the
                  only cells for which information is gathered/converted

            We require that :arg:`mesh` have co-dimesnion
            of 0 or 1.
            Moreover, if :arg:`mesh` is a 2-surface embedded in 3-space,
            we _require_ that :function:`init_cell_orientations`
            has been called already.
        """
        top = mesh.topological

        # {{{ Check input
        if not isinstance(top, MeshTopology):
            raise TypeError(":arg:`mesh` must be of type "
                            ":class:`firedrake.mesh.MeshTopology` or "
                            ":class:`firedrake.mesh.MeshGeometry`")
        # }}}

        super(MeshTopologyAnalog, self).__init__(top)

        # Ensure has simplex-type elements
        if not top.ufl_cell().is_simplex():
            raise ValueError("mesh must have simplex type elements, "
                             "%s is not a simplex" % (mesh.ufl_cell()))

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        if self.cell_dimension() not in supported_dims:
            raise ValueError("Cell dimension is %s. Cell dimension must be one of"
                             "range %s" % (self.cell_dimension(), supported_dims))

        self._nodal_adjacency = None
        self.icell_to_fd = cells_to_use  # Map cell index -> fd cell index
        self.fd_to_icell = None          # Map fd cell index -> cell index
        if self.icell_to_fd is not None:
            assert np.unique(self.icell_to_fd).shape == self.icell_to_fd.shape
            self.fd_to_icell = dict(zip(self.icell_to_fd,
                                         np.arange(self.icell_to_fd.shape[0],
                                                   dtype=np.int32)
                                        ))

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
        if self.icell_to_fd is None:
            num_cells = self.analog().num_cells()
        else:
            num_cells = self.icell_to_fd.shape[0]

        return num_cells

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
            bdy_tags += list(unique_markers)

        return tuple(bdy_tags)

    def nodal_adjacency(self):
        """
            Returns a :class:`meshmode.mesh.NodalAdjacency` object
            representing the nodal adjacency of this mesh
        """
        if self._nodal_adjacency is None:
            plex = self.analog()._plex

            cStart, cEnd = plex.getHeightStratum(0)
            vStart, vEnd = plex.getDepthStratum(0)

            to_fd_id = np.vectorize(self.analog()._cell_numbering.getOffset)(
                np.arange(cStart, cEnd, dtype=np.int32))

            element_to_neighbors = {}
            verts_checked = set()  # dmplex ids of vertex checked

            # If using all cells, loop over them all
            if self.icell_to_fd is None:
                range_ = range(cStart, cEnd)
            # Otherwise, just the ones you're using
            else:
                isin = np.isin(to_fd_id, self.icell_to_fd)
                range_ = np.arange(cStart, cEnd, dtype=np.int32)[isin]

            # For each cell
            for cell_id in range_:
                # For each vertex touching the cell (that haven't already seen)
                for vert_id in plex.getTransitiveClosure(cell_id)[0]:
                    if vStart <= vert_id < vEnd and vert_id not in verts_checked:
                        verts_checked.add(vert_id)
                        cells = []
                        # Record all cells touching that vertex
                        support = plex.getTransitiveClosure(vert_id,
                                                            useCone=False)[0]
                        for other_cell_id in support:
                            if cStart <= other_cell_id < cEnd:
                                cells.append(to_fd_id[other_cell_id - cStart])

                        # If only using some cells, clean out extraneous ones
                        # and relabel them to new id
                        cells = set(cells)
                        if self.fd_to_icell is not None:
                            cells = set([self.fd_to_icell[fd_ndx]
                                         for fd_ndx in cells
                                         if fd_ndx in self.fd_to_icell])

                        # mark cells as neighbors
                        for cell_one in cells:
                            element_to_neighbors.setdefault(cell_one, set())
                            element_to_neighbors[cell_one] |= cells

            # Create neighbors_starts and neighbors
            neighbors = []
            neighbors_starts = np.zeros(self.nelements() + 1, dtype=np.int32)
            for iel in range(len(element_to_neighbors)):
                elt_neighbors = element_to_neighbors[iel]
                neighbors += list(elt_neighbors)
                neighbors_starts[iel+1] = len(neighbors)

            neighbors = np.array(neighbors, dtype=np.int32)

            self._nodal_adjacency = NodalAdjacency(neighbors_starts=neighbors_starts,
                                                   neighbors=neighbors)
        return self._nodal_adjacency


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

    def __init__(self, mesh, coordinates_analog, normals=None, no_normals_warn=True):
        """
            :arg mesh: A :mod:`firedrake` :class:`MeshGeometry`.
                We require that :arg:`mesh` have co-dimesnion
                of 0 or 1.
                Moreover, if :arg:`mesh` is a 2-surface embedded in 3-space,
                we _require_ that :function:`init_cell_orientations`
                has been called already.

            :arg coordinates_analog: A :class:`CoordinatelessFunctionSpaceAnalog`
                                      to use, represents the coordinates

            For other args see :meth:`orientations`
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
        from fd2mm.function import CoordinatelessFunctionAnalog
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

        self._normals = normals
        self._no_normals_warn = no_normals_warn

        # To be computed later
        self._vertex_indices = None
        self._vertices = None
        self._nodes = None
        self._group = None
        self._orient = None
        self._facial_adjacency_groups = None
        self._meshmode_mesh = None

        def callback(cl_ctx):
            """
                Finish initialization
            """
            from fd2mm.functionspaceimpl import WithGeometryAnalog
            from firedrake import Function
            from fd2mm.function import FunctionAnalog

            coordinates_fs = self.analog().coordinates.function_space()
            coordinates_fs_a = self._coordinates_a.function_space_a()

            V_a = WithGeometryAnalog(cl_ctx, coordinates_fs, coordinates_fs_a, self)
            f = Function(V_a.analog(), val=self._coordinates_a.analog())
            self._coordinates_function_a = FunctionAnalog(f, V_a)

            del self._callback

        self._callback = callback

    def initialized(self):
        return not hasattr(self, '_callback')

    def init(self, cl_ctx):
        if not self.initialized():
            self._callback(cl_ctx)

    def __getattr__(self, attr):
        """
        Done like :class:`firedrake.function.MeshGeometry`
        """
        return getattr(self._topology_a, attr)

    @property
    def coordinates_a(self):
        """
            Return coordinates as a function

            PRECONDITION: Have called
        """
        try:
            return self._coordinates_function_a
        except AttributeError:
            raise AttributeError("No coordinates function, have you finished"
                                 " initializing this analog?"
                                 " (i.e. have you called :meth:`init`")

    def _compute_vertex_indices_and_vertices(self):
        if self._vertex_indices is None:
            finat_element_a = self.coordinates_a.function_space_a().finat_element_a

            # Convert cell node list of mesh to vertex list
            unit_vertex_indices = finat_element_a.unit_vertex_indices()
            cfspace = self.analog().coordinates.function_space()
            if self.icell_to_fd is not None:
                cell_node_list = cfspace.cell_node_list[self.icell_to_fd]
            else:
                cell_node_list = cfspace.cell_node_list

            vertex_indices = cell_node_list[:, unit_vertex_indices]

            # Get maps newnumbering->old and old->new (new numbering comes
            #                                          from removing the non-vertex
            #                                          nodes)
            vert_ndx_to_fd_ndx = np.unique(vertex_indices.flatten())
            fd_ndx_to_vert_ndx = dict(zip(vert_ndx_to_fd_ndx,
                                          np.arange(vert_ndx_to_fd_ndx.shape[0],
                                                    dtype=np.int32)
                                          ))
            # Get vertices array
            vertices = np.real(
                self.analog().coordinates.dat.data[vert_ndx_to_fd_ndx])

            #:mod:`meshmode` wants shape to be [ambient_dim][nvertices]
            if len(vertices.shape) == 1:
                # 1 dim case, (note we're about to transpose)
                vertices = vertices.reshape(vertices.shape[0], 1)
            vertices = vertices.T.copy()

            # Use new numbering on vertex indices
            vertex_indices = np.vectorize(fd_ndx_to_vert_ndx.get)(vertex_indices)

            # store vertex indices and vertices
            self._vertex_indices = vertex_indices
            self._vertices = vertices

    def vertex_indices(self):
        self._compute_vertex_indices_and_vertices()
        return self._vertex_indices

    def vertices(self):
        self._compute_vertex_indices_and_vertices()
        return self._vertices

    def nodes(self):
        if self._nodes is None:
            coords = self.analog().coordinates.dat.data
            cfspace = self.analog().coordinates.function_space()

            if self.icell_to_fd is not None:
                cell_node_list = cfspace.cell_node_list[self.icell_to_fd]
            else:
                cell_node_list = cfspace.cell_node_list
            self._nodes = np.real(coords[cell_node_list])

            # reshape for 1D so that [nelements][nunit_nodes][dim]
            if len(self._nodes.shape) != 3:
                self._nodes = np.reshape(self._nodes, self._nodes.shape + (1,))

            # Change shape to [dim][nelements][nunit_nodes]
            self._nodes = np.transpose(self._nodes, (2, 0, 1))

        return self._nodes

    def group(self):
        if self._group is None:
            from meshmode.mesh import SimplexElementGroup
            from meshmode.mesh.processing import flip_simplex_element_group

            finat_element_a = self.coordinates_a.function_space_a().finat_element_a

            # IMPORTANT that set :attr:`_group` because
            # :meth:`orientations` may call :meth:`group`
            self._group = SimplexElementGroup(
                finat_element_a.analog().degree,
                self.vertex_indices(),
                self.nodes(),
                dim=self.cell_dimension(),
                unit_nodes=finat_element_a.unit_nodes())

            self._group = flip_simplex_element_group(self.vertices(), self._group,
                                                     self.orientations() < 0)

        return self._group

    def orientations(self):
        """
            Return the orientations of the mesh elements:
            an array, the *i*th element is > 0 if the *ith* element
            is positively oriented, < 0 if negatively oriented

            :arg normals: _Only_ used if :arg:`mesh` is a 1-surface
                embedded in 2-space. In this case,
                - If *None* then
                  all elements are assumed to be positively oriented.
                - Else, should be a list/array whose *i*th entry
                  is the normal for the *i*th element (*i*th
                  in :arg:`mesh`*.coordinate.function_space()*'s
                  :attribute:`cell_node_list`)

            :arg no_normals_warn: If *True*, raises a warning
                if :arg:`mesh` is a 1-surface embedded in 2-space
                and :arg:`normals` is *None*.
        """
        if self._orient is None:
            # compute orientations
            tdim = self.analog().topological_dimension()
            gdim = self.analog().geometric_dimension()

            orient = None
            if gdim == tdim:
                # We use :mod:`meshmode` to check our orientations
                from meshmode.mesh.processing import \
                    find_volume_mesh_element_group_orientation

                orient = \
                    find_volume_mesh_element_group_orientation(self.vertices(),
                                                               self.group())

            if tdim == 1 and gdim == 2:
                # In this case we have a 1-surface embedded in 2-space
                orient = np.ones(self.nelements())
                if self._normals:
                    for i, (normal, vertices) in enumerate(zip(
                            np.array(self._normals), self.vertices())):
                        if np.cross(normal, vertices) < 0:
                            orient[i] = -1.0
                elif self._no_normals_warn:
                    warn("Assuming all elements are positively-oriented.")

            elif tdim == 2 and gdim == 3:
                # In this case we have a 2-surface embedded in 3-space
                orient = self.analog().cell_orientations().dat.data
                r"""
                    Convert (0 \implies negative, 1 \implies positive) to
                    (-1 \implies negative, 1 \implies positive)
                """
                orient *= 2
                orient -= np.ones(orient.shape, dtype=orient.dtype)

            self._orient = orient
            #Make sure the mesh fell into one of the cases
            """
              NOTE : This should be guaranteed by previous checks,
                     but is here anyway in case of future development.
            """
            assert self._orient is not None

        return self._orient

    def face_vertex_indices_to_tags(self):
        """
            Return a :mod:`meshmode` list of :class:`FacialAdjacencyGroups`
            as used in the construction of a :mod:`meshmode` :class:`Mesh`
        """
        finat_element = self.analog().coordinates.function_space().finat_element
        exterior_facets = self.analog().exterior_facets

        # fvi_to_tags maps frozenset(vertex indices) to tags
        fvi_to_tags = {}
        # maps faces to local vertex indices
        connectivity = finat_element.cell.connectivity[(self.cell_dimension()-1, 0)]

        for i, (icells, ifacs) in enumerate(zip(exterior_facets.facet_cell,
                                                exterior_facets.local_facet_number)):
            for icell, ifac in zip(icells, ifacs):
                # If necessary, convert to new cell numbering
                if self.fd_to_icell is not None:
                    if icell not in self.fd_to_icell:
                        continue
                    else:
                        icell = self.fd_to_icell[icell]

                # record face vertex indices to tag map
                cell_vertices = self.vertex_indices()[icell]
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
        # {{{ Compute facial adjacency groups if not already done

        if self._facial_adjacency_groups is None:
            from meshmode.mesh import _compute_facial_adjacency_from_vertices

            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                [self.group()],
                self.bdy_tags(),
                np.int32, np.int8,
                face_vertex_indices_to_tags=self.face_vertex_indices_to_tags())

        # }}}

        return self._facial_adjacency_groups

    def meshmode_mesh(self):
        """
        PRECONDITION: Have called :meth:`init`
        """
        if self._meshmode_mesh is None:
            assert self.initialized(), \
                "Must call :meth:`init` before :meth:`meshmode_mesh`"

            from meshmode.mesh import Mesh
            self._meshmode_mesh = \
                Mesh(self.vertices(), [self.group()],
                     boundary_tags=self.bdy_tags(),
                     nodal_adjacency=self.nodal_adjacency(),
                     facial_adjacency_groups=self.facial_adjacency_groups())

        return self._meshmode_mesh


def _compute_cells_near_bdy(mesh, bdy_id):
    """
        Returns an array of the cell ids with >= 1 vertex on the
        given bdy_id
    """
    cfspace = mesh.coordinates.function_space()
    cell_node_list = cfspace.cell_node_list

    boundary_nodes = cfspace.boundary_nodes(bdy_id, 'topological')
    # Reduce along each cell: Is a vertex of the cell in boundary nodes?
    cell_is_near_bdy = np.any(np.isin(cell_node_list, boundary_nodes), axis=1)

    return np.arange(cell_node_list.shape[0], dtype=np.int32)[cell_is_near_bdy]


def MeshAnalog(mesh, near_bdy=None):
    coords_fspace = mesh.coordinates.function_space()
    cells_to_use = None
    if near_bdy is not None:
        cells_to_use = _compute_cells_near_bdy(mesh, near_bdy)

    topology_a = MeshTopologyAnalog(mesh, cells_to_use=cells_to_use)
    finat_elt_a = FinatElementAnalog(coords_fspace.finat_element)

    from fd2mm.functionspaceimpl import FunctionSpaceAnalog
    from fd2mm.function import CoordinatelessFunctionAnalog

    coords_fspace_a = FunctionSpaceAnalog(coords_fspace,
                                          topology_a,
                                          finat_elt_a)
    coordinates_analog = CoordinatelessFunctionAnalog(mesh.coordinates,
                                                      coords_fspace_a)

    return MeshGeometryAnalog(mesh, coordinates_analog)
