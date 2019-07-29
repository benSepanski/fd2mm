from warnings import warn
import numpy as np

from meshmode.mesh import BTAG_ALL
from firedrake_to_pytential.analogs import Analog


class MeshAnalog(Analog):
    """
        This takes a :mod:`firedrake` :class:`MeshGeometry`
        and converts its data so that :mod:`meshmode` can handle it.

        NOTE: This an analog of a :mod:`firedrake` mesh, but NOT
              an analog of a :mod:`meshmode` mesh. In particular,
              it doesn't store node information, just vertex
              information. If you are wanting a :mod:`meshmode` mesh,
              you really need a :class:`FunctionSpaceAnalog`.
    """

    def __init__(self, mesh, normals=None, no_normals_warn=True):
        """
            :arg mesh: A :mod:`firedrake` :class:`MeshGeometry`.
                We require that :arg:`mesh` have co-dimesnion
                of 0 or 1.
                Moreover, if :arg:`mesh` is a 2-surface embedded in 3-space,
                we _require_ that :function:`init_cell_orientations`
                has been called already.

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
        super(MeshAnalog, self).__init__(mesh)

        # {{{ Make sure input data is valid

        # Ensure is not a topological mesh
        assert mesh.topological != mesh

        self._tdim = mesh.topological_dimension()
        self._gdim = mesh.geometric_dimension()

        # Ensure has simplex-type elements
        if not mesh.ufl_cell().is_simplex():
            raise ValueError("mesh must have simplex type elements, "
                             "%s is not a simplex" % (mesh.ufl_cell()))

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        for dim, name in zip([self._tdim, self._gdim], ["topological", "geometric"]):
            if dim not in supported_dims:
                raise ValueError("%s dimension is %s. %s dimension must be one of"
                                 "range %s" % (name, dim, name, supported_dims))

        # Raise warning if co-dimension is not 0 or 1
        co_dimension = self._gdim - self._tdim
        if co_dimension not in [0, 1]:
            raise ValueError("Codimension is %s, but must be 0 or 1." %
                             (co_dimension))

        # Store normal information
        self.normals = normals
        self.no_normals_warn = no_normals_warn

        # Create attributes to be computed later
        self._vertices = None                 # See method vertices()
        self._vertex_indices = None           # See method vertex_indices()
        self._orient = None                   # See method orientation()
        self._facial_adjacency_groups = None
        self._boundary_tags = None

        # A :mod:`meshmode` :class:`MeshElementGroup` used to construct
        # orientations and facial adjacency
        self._group = None

    def topological_dimension(self):
        """
            Return topological dimension of what this object is an
            analog of
        """
        return self._tdim

    def geometric_dimension(self):
        """
            Return geometric dimension of what this object is an
            analog of
        """
        return self._gdim

    def vertices(self):
        """
        An array holding the coordinates of the vertices of shape
        (geometric dim, nvertices)
        """
        if self._vertices is None:
            self._vertices = np.real(self.analog().coordinates.dat.data)

            #:mod:`meshmode` wants [ambient_dim][nvertices], but for now we
            #write it as [geometric dim][nvertices]
            self._vertices = self._vertices.T.copy()

        return self._vertices

    def vertex_indices(self):
        """
            (analog to mesh.coordinates.function_space().cell_node_list)

            <from :mod:`meshmode` docs:>
            An array of (nelements, ref_element.nvertices)
            of (mesh-wide) vertex indices
        """
        if self._vertex_indices is None:
            self._vertex_indices = np.copy(
                self.analog().coordinates.function_space().cell_node_list)

        return self._vertex_indices

    def orientations(self):
        """
            Return the orientations of the mesh elements:
            an array, the *i*th element is > 0 if the *ith* element
            is positively oriented, < 0 if negatively oriented
        """
        # TODO: This is probably inefficient design... but some of the
        #       computation needs a :mod:`meshmode` group
        #       right now.
        from meshmode.mesh.generation import make_group_from_vertices
        self._group = make_group_from_vertices(self.vertices(),
                                               self.vertex_indices(), 1)

        # {{{ Compute the orientations if necessary
        if self._orient is None:

            if self._gdim == self._tdim:
                # We use :mod:`meshmode` to check our orientations
                from meshmode.mesh.processing import \
                    find_volume_mesh_element_group_orientation

                self._orient = \
                    find_volume_mesh_element_group_orientation(self.vertices(),
                                                               self._group)

            if self._tdim == 1 and self._gdim == 2:
                # In this case we have a 1-surface embedded in 2-space
                self._orient = np.ones(self.vertex_indices().shape[0])
                if self.normals:
                    for i, (normal, vertices) in enumerate(zip(
                            np.array(self.normals), self.vertices())):
                        if np.cross(normal, vertices) < 0:
                            self._orient[i] = -1.0
                elif self.no_normals_warn:
                    warn("Assuming all elements are positively-oriented.")

            elif self._tdim == 2 and self._gdim == 3:
                # In this case we have a 2-surface embedded in 3-space
                r"""
                    Convert (0 \implies negative, 1 \implies positive) to
                    (-1 \implies negative, 1 \implies positive)
                """
                self._orient *= 2
                self._orient -= np.ones(self._orient.shape)

            #Make sure the mesh fell into one of the above cases
            """
              NOTE : This should be guaranteed by previous checks,
                     but is here anyway in case of future development.

                     In general, I will raise exceptions for errors
                     I expect a user to encounter, and assert for
                     errors I consider more likely on the development
                     side.
            """
            assert self._orient is not None

        # }}}

        return self._orient

    def _get_cell_vertices(self, icell):
        """
        :arg icell: The firedrake index of a cell

        This is its own function so that it can be overloaded by
        descendants of this class, which may have different
        cell sets/orderings.

        Inheritors should return *None* if icell is a bad index
        """
        return self.vertex_indices()[icell]

    def facial_adjacency_groups(self):
        """
            Return a :mod:`meshmode` list of :class:`FacialAdjacencyGroups`
            as used in the construction of a :mod:`meshmode` :class:`Mesh`
        """

        # {{{ Compute facial adjacency groups if not already done

        if self._facial_adjacency_groups is None:
            # Create a group for later use
            orient = self.orientations()

            from meshmode.mesh.processing import flip_simplex_element_group
            self._group = flip_simplex_element_group(self.vertices(),
                                                     self._group,
                                                     orient < 0)
            groups = [self._group]

            # {{{ Get boundary data

            # fvi_to_tags maps frozenset(vertex indices) to tags
            fvi_to_tags = {}
            # maps faces to local vertex indices
            connectivity = \
                self.analog().coordinates.function_space().finat_element.cell.\
                connectivity[(self._tdim - 1, 0)]

            efacets = self.analog().exterior_facets

            if efacets.facet_cell.size == 0 and self._tdim >= self._gdim:
                warn("No exterior facets listed in"
                     " <mesh>.exterior_facets.facet_cell. In particular, NO BOUNDARY"
                     " information is tagged.")

            for i, (icells, ifacs) in enumerate(zip(
                    efacets.facet_cell, efacets.local_facet_number)):
                for icell, ifac in zip(icells, ifacs):

                    cell_vertices = self._get_cell_vertices(icell)
                    if cell_vertices is None:
                        # If cell index is not relevant, continue (this is for
                        #                                          inheritors of
                        #                                          this class)
                        continue

                    # record face vertex indices to tag map
                    facet_indices = connectivity[ifac]
                    fvi = frozenset(cell_vertices[list(facet_indices)])
                    fvi_to_tags.setdefault(fvi, [])
                    fvi_to_tags[fvi].append(efacets.markers[i])

            # }}}

            from meshmode.mesh import _compute_facial_adjacency_from_vertices
            """
                NOTE : This relies HEAVILY on the fact that elements are *not*
                       reordered at any time, and that *_vertex_indices*
                       are also not reordered.
            """
            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                groups,
                self.boundary_tags(),
                np.int32, np.int8,
                face_vertex_indices_to_tags=fvi_to_tags)

        # }}}

        return self._facial_adjacency_groups

    def boundary_tags(self):
        """
            Return a tuple of boundary tags as requested in
            the construction of a :mod:`meshmode` :class:`Mesh`
        """
        # Compute boundary tags if needed
        if self._boundary_tags is None:
            from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
            self._boundary_tags = [BTAG_ALL, BTAG_REALLY_ALL]

            efacets = self.analog().exterior_facets
            if efacets.unique_markers is not None:
                for tag in efacets.unique_markers:
                    self._boundary_tags.append(tag)
            self._boundary_tags = tuple(self._boundary_tags)

        return self._boundary_tags


class MeshAnalogNearBoundary(MeshAnalog):
    """
        As :class:`MeshAnalog`, but ONLY converts data near
        :attr:`near_bdy`

        A cell is considered "near" :attr:`near_bdy` if
        it has at least one vertex on the boundary
    """

    def __init__(self, mesh, near_bdy, normals=None, no_normals_warn=True):
        """
            :arg near_bdy: An *int* which is an exterior boundary marker,
                           for all boundaries use either the string
                           'on_boundary' or the :mod:`meshmode` class
                           :class:`mesh.BTAG_ALL`
        """
        # verify the ids are valid markers, store near_bdy,
        if not isinstance(near_bdy, int) \
                and near_bdy != 'on_boundary' and near_bdy != BTAG_ALL:
            raise TypeError("near_bdy must be an int, 'on_boundary', or"
                            "meshmode.mesh.BTAG_ALL")

        valid_bdy_ids = mesh.exterior_facets.unique_markers
        assert near_bdy in valid_bdy_ids

        if near_bdy == BTAG_ALL:
            self._near_bdy = 'on_boundary'
        else:
            self._near_bdy = near_bdy

        # see method `_compute_near_bdy` for both of these
        self._verts_near_bdy = None
        self._cells_near_bdy = None

        # dict mapping firedrake cell index to cell index in this object
        self._fd_cell_index_to_new = None

        super(MeshAnalogNearBoundary, self).__init__(mesh, normals=normals,
                                                     no_normals_warn=no_normals_warn)

    def contains_bdy(self, bdy_id):
        """
            Return *True* iff contains the given bdy_id. :arg:`bdy_id`
            should be as in :arg:`near_bdy` in :meth:`__init__`.
        """
        # If :attr:`_near_bdy` is a boundary id not equal to :kwarg:`near_bdy`,
        # this is not an analog
        if isinstance(self._near_bdy, int) and self._near_bdy != bdy_id:
            return False

        # Otherwise if :attr:`_near_bdy` is the whole boundary and
        # near_bdy is not a valid boundary id, this is not an analog
        elif self._near_bdy == 'on_boundary':
            if bdy_id not in self.analog().exterior_facets.unique_markers:
                return False

        return True

    def is_analog(self, obj, **kwargs):
        """
            :kwarg near_bdy: As in construction of a :class:`MeshAnalogNearBoundary`
                             defaults to *None*

            Return whether or not this object is an analog for the
            given object and near_bdy
        """
        near_bdy = kwargs.get('near_bdy', None)
        if near_bdy == BTAG_ALL:
            near_bdy = 'on_boundary'

        return self.contains_bdy(near_bdy) and \
            super(MeshAnalogNearBoundary, self).is_analog(obj)

    def _compute_near_bdy(self):
        """
            Creates *np.ndarray* of indexes :attr:`_cells_near_bdy`,
            those cells with at least one vertex on a boundary of
            :attr:`_near_bdy`, and also an *np.ndarray* of indexes
            :attr:`_verts_near_bdy`, all the vertex indices of
            vertices on a cell near a boundary.
        """
        # Compute if necessary (check both just to be safe)
        if self._verts_near_bdy is None or self._cells_near_bdy is None:

            # {{{ Get all the vertex indices of vertices on the boundary
            coord_fspace = self.analog().coordinates.function_space()

            verts_on_bdy = coord_fspace.boundary_nodes(self._near_bdy, 'topological')

            # }}}

            # FIXME : This still might be slow, having to iterate over all the
            #         cells
            # Get all vertices of cells with at least one vertex on a near bdy
            # Also record the cells
            mesh = coord_fspace.mesh()
            plex = mesh._plex
            # First dmplex vert number to 1+last
            vStart, vEnd = plex.getDepthStratum(0)
            # First dmplex cell number to 1+last
            cStart, cEnd = plex.getHeightStratum(0)

            # firedrake vertexes ordered by their dmplex ordering
            dm_relative_vert_order_to_fd = np.vectorize(
                mesh._vertex_numbering.getOffset)(np.arange(vStart, vEnd))
            # maps firedrake vertex to dmplex vertex
            fd_vert_to_dm = np.argsort(dm_relative_vert_order_to_fd) + vStart

            # Compute actual cells and verts near the given boundary
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

            # Convert to list, preserving relative ordering
            self._verts_near_bdy = np.array(list(verts_near_bdy),
                                            dtype=np.int32)
            # Store which cells are near boundary
            self._cells_near_bdy = np.array(list(cells_near_bdy),
                                            dtype=np.int32)

        return

    def verts_near_bdy(self):
        """
            see _compute_near_bdy
        """
        self._compute_near_bdy()
        return self._verts_near_bdy

    def cells_near_bdy(self):
        """
            see _compute_near_bdy
        """
        self._compute_near_bdy()
        return self._cells_near_bdy

    def vertices(self):
        """
        As :class:`MeshAnalog`.:meth:`vertices, except that only gives
        the vertices contained in an element which has at least one
        vertex on the near boundary.
        """
        if self._vertices is None:
            self._vertices = np.real(self.analog().coordinates.dat.data)

            # Only use vertices of cells near the given boundary
            self._vertices = self._vertices[self.verts_near_bdy()]

            #:mod:`meshmode` wants [ambient_dim][nvertices], but for now we
            #write it as [geometric dim][nvertices]
            self._vertices = self._vertices.T.copy()

        return self._vertices

    def vertex_indices(self):
        """
            As :class:`MeshAnalog`.:meth:`vertex_indices`, except
            this will only give the vertex indices of cells which have at least one
            vertex on the near boundary.
        """
        if self._vertex_indices is None:
            self._vertex_indices = np.copy(
                self.analog().coordinates.function_space().cell_node_list)

            # Maps firedrake vertex index to new index
            verts_near_bdy_inv = dict(zip(
                self.verts_near_bdy(),
                np.arange(self.verts_near_bdy().shape[0], dtype=np.int32)
                ))

            self._vertex_indices = self._vertex_indices[self.cells_near_bdy()]
            # Change old vertex index to new vertex index
            self._vertex_indices = np.vectorize(verts_near_bdy_inv.get
                                                )(self._vertex_indices)

        return self._vertex_indices

    def _get_cell_vertices(self, icell):
        """
        As in :class:`MeshAnalog`
        """
        if self._fd_cell_index_to_new is None:
            # inverse to :attr:`_cells_near_bdy`
            self._fd_cell_index_to_new = dict(zip(
                self.cells_near_bdy(),
                np.arange(self.cells_near_bdy().shape[0], dtype=np.int32)
                ))

        # If not a valid cell index, return *None*
        if icell not in self._fd_cell_index_to_new:
            return None

        # Else, return the cell vertices
        new_index = self._fd_cell_index_to_new[icell]
        return self.vertex_indices()[new_index]
