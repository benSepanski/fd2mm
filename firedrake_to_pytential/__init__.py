import pyopencl as cl
import six

import numpy as np
from numpy import linalg as la

import firedrake as fd

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
from modepy import tools

from pytential.qbx import QBXLayerPotentialSource
from warnings import warn


"""
    Analogs are containers which hold the information
    needed to convert between :mod:`firedrake` and
    :mod:`meshmode`.
"""


class MeshAnalog:
    """
        This takes a :mod:`firedrake` :class:`MeshGeometry`
        and converts its data into :mod:`meshmode` format.

        .. attribute::  _tdim

            topological dimension.

        .. attribute::  _gdim

            geometric dimension

        .. attribute::  _vertices

            vertex coordinates (analog to mesh.coordinates.dat.data)

        .. attribute::  _vertex_indices

            vertex indices (analog to
                mesh.coordinates.function-space().cell_node_list)

        .. attribute::  _orient

            An array, the *i*th element is > 0 if the *ith* element
            is positively oriented, < 0 if negatively oriented

        .. attribute:: _facial_adjacency_groups

            describes facial adjacency

        .. attribute::  _mesh

            The :mod:`firedrake` :class:`mesh` this object is an analog to.
            Must have simplex elements.
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

        # }}}

        # {{{ Get coordinates (vertices in meshmode language)

        # Get coordinates of vertices
        self._vertices = mesh.coordinates.dat.data.astype(np.float64)
        """
        :mod:`meshmode` wants [ambient_dim][nvertices], but for now we
        write it as [geometric dim][nvertices]
        """
        self._vertices = self._vertices.T.copy()

        # get vertex indices
        vertices_fn_space = mesh.coordinates.function_space()
        """
        <from :mod:`meshmode` docs:>
        An array of (nelements, ref_element.nvertices)
        of (mesh-wide) vertex indices
        """
        self._vertex_indices = np.copy(vertices_fn_space.cell_node_list)

        # }}}

        # TODO: This is probably inefficient design... but some of the
        #       computation needs a :mod:`meshmode` group
        #       right now.
        from meshmode.mesh.generation import make_group_from_vertices
        group = make_group_from_vertices(self._vertices,
                                                 self._vertex_indices, 1)

        # {{{ Compute the orientations

        self._orient = None
        if self._gdim == self._tdim:
            # We use :mod:`meshmode` to check our orientations
            from meshmode.mesh.processing import \
                find_volume_mesh_element_group_orientation

            self._orient = \
                find_volume_mesh_element_group_orientation(self._vertices,
                                                           group)

        if self._tdim == 1 and self._gdim == 2:
            # In this case we have a 1-surface embedded in 2-space
            self._orient = np.ones(self._vertex_indices.shape[0])
            if normals:
                for i, (normal, vertices) in enumerate(zip(
                        np.array(normals), self._vertices)):
                    if np.cross(normal, vertices) < 0:
                        self._orient[i] = -1.0
            elif no_normals_warn:
                warn("Assuming all elements are positively-oriented.")

        elif self._tdim == 2 and self._gdim == 3:
            # In this case we have a 2-surface embedded in 3-space
            self._orient = mesh.cell_orientations().dat.data.astype(np.float64)
            """
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

        # Create a group for later use
        from meshmode.mesh.processing import flip_simplex_element_group
        group = flip_simplex_element_group(self._vertices,
                                           group,
                                           self._orient < 0)
        groups = [group]

        # {{{ Get boundary data

        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
        boundary_tags = [BTAG_ALL, BTAG_REALLY_ALL]

        efacets = mesh.exterior_facets
        if efacets.unique_markers is not None:
            for tag in efacets.unique_markers:
                boundary_tags.append(tag)
        boundary_tags = tuple(boundary_tags)

        # fvi_to_tags maps frozenset(vertex indices) to tags
        fvi_to_tags = {}
        connectivity = vertices_fn_space.finat_element.cell.\
            connectivity[(self._dim - 1, 0)]  # maps faces to local vertex indices

        for i, (icell, ifac) in enumerate(zip(
                efacets.facet_cell, efacets.local_facet_number)):
            # unpack arguments
            ifac, = ifac
            icell, = icell
            # record face vertex indices to tag map
            facet_indices = connectivity[ifac]
            fvi = frozenset(self._vertices[icell]
                            [list(facet_indices)])
            fvi_to_tags.setdefault(fvi, [])
            fvi_to_tags[fvi].append(efacets.markers[i])

        from meshmode.mesh import _compute_facial_adjacency_from_vertices
        """
            NOTE : This relies HEAVILY on the fact that elements are *not*
                   reordered at any time, and that *_vertex_indices*
                   are also not reordered.
        """
        self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
            groups,
            boundary_tags,
            np.int32, np.int8,
            face_vertex_indices_to_tags=fvi_to_tags)
        # }}}

        self._boundary_tags = boundary_tags
        self._mesh = mesh

    def topological_dimension(self):
        return self._tdim

    def geometric_dimension(self):
        return self._gdim


def _get_affine_mapping(v, w):
    """
    Returns (A, b),
    a matrix A and vector b which maps the *i*th vector in :arg:`v`
    to the *i*th vector in :arg:`w` by
    Avi + b -> wi

    :arg v: An np.array of *n* vectors of dimension *d*
    :arg w: An np.array of *n* vectors of dimension *d*
    """
    assert v.shape[0] == w.shape[0]

    if v.shape[0] == 1:
        A = np.eye(v.shape[0])
        b = v[0] - w[0]
    else:
        v_span_vects = v[:, 1:] - v[:, 0, np.newaxis]
        w_span_vects = w[:, 1:] - w[:, 0, np.newaxis]
        A = np.linalg.solve(v_span_vects, w_span_vects)
        b = np.matmul(A, -v[:, 0]) + w[:, 0]
    return A, b


class SimplexCellAnalog:
    """
        cell analog, for simplices only
    """
    def __init__(self, cell):

        reference_vertices = np.array(cell.vertices)

        dim = reference_vertices.shape[0] - 1
        self._unit_vertices = tools.unit_vertices(dim)

        # Maps firedrake reference nodes to :mod:`meshmode`
        # unit nodes
        self._A, self._b = _get_affine_mapping(reference_vertices,
                                               self._unit_vertices)
        self._cell = cell

    def make_points(self, dim, entity_id, order):
        """
            as called by a cell in firedrake, but converted
            to :mod:`modepy` unit nodes
        """
        points = self._cell.make_points(dim, entity_id, order)
        points = np.array(points)
        return np.matmul(self._A, points) + self._b


class DGFunctionSpaceAnalog:
    """
    """

    def __init__(self, function_space, mesh_analog=None, cell_analog=None):

        if function_space.ufl_element().family() != 'Discontinuous Lagrange':
            raise ValueError("Must use Discontinuous Lagrange elements")

        if mesh_analog is None:
            mesh_analog = MeshAnalog(function_space.mesh())

        if cell_analog is None:
            cell_analog = SimplexCellAnalog(function_space.ufl_element().cell)

        self._unit_nodes = None
        self._flip_matrix = None
        self._nodes = None
        self._meshmode_mesh = None

        # Base ufl element
        self._ufl_base = function_space.ufl_element()
        if function_space.shape:
            if len(set(function_space.sub_elements())) > 1:
                raise ValueError("No support for mixed function spaces")
            self._ufl_base = function_space.sub_elements()[0]

        self._mesh_analog = mesh_analog
        self._cell_analog = cell_analog

    def _get_unit_nodes(self):
        if self._unit_nodes is None:
            fe = self._function_space.finat_element
            order = self._order
            node_nr_to_coords = {}

            # {{{ Get unit nodes (in :mod:`meshmode` coordinates)
            for dim, element_nrs in six.iteritems(fe.entity_support_dofs()):
                for element_nr, node_list in six.iteritems(element_nrs):
                    pts_on_element = self._cell_analog.make_points(dim, element_nr,
                                                                   order)
                    i = 0
                    for node_nr in node_list:
                        if node_nr not in node_nr_to_coords:
                            node_nr_to_coords[node_nr] = pts_on_element[i]
                            i += 1
            # }}}

            # Convert unit_nodes to array, then change to (dim, nunit_nodes)
            # from (nunit_nodes, dim)
            unit_nodes = np.array([node_nr_to_coords[i] for i in
                                   range(len(node_nr_to_coords))])
            self._unit_nodes = unit_nodes.T.copy()

        return self._unit_nodes

    def _get_flip_matrix(self):
        if self._flip_matrix is None:
            grp = self.mesh_map['domain'].groups[0]

            from modepy.tools import barycentric_to_unit, unit_to_barycentric

            from meshmode.mesh import SimplexElementGroup

            if not isinstance(grp, SimplexElementGroup):
                raise NotImplementedError("flips only supported on "
                        "exclusively SimplexElementGroup-based meshes")

            # Generate a resampling matrix that corresponds to the
            # first two barycentric coordinates being swapped.

            bary_unit_nodes = unit_to_barycentric(grp.unit_nodes)

            flipped_bary_unit_nodes = bary_unit_nodes.copy()
            flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
            flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
            flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

            from modepy import resampling_matrix, simplex_best_available_basis
            flip_matrix = resampling_matrix(
                    simplex_best_available_basis(grp.dim, grp.order),
                    flipped_unit_nodes, grp.unit_nodes)

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                    np.dot(flip_matrix, flip_matrix)
                    - np.eye(len(flip_matrix))) < self._thresh

            self._flip_matrix = flip_matrix

        return self._flip_matrix

    def function_space_is_valid(self, function_space):
        ufl_base = function_space.ufl_element()
        if function_space.shape:
            if len(set(function_space.sub_elements())) > 1:
                return False
            ufl_base = function_space.sub_elements()[0]

        if ufl_base != self._ufl_base:
            return False

        if self._mesh_analog._mesh != function_space.mesh():
            return False

        return True

    def reorder_nodes(self, nodes, firedrake_to_meshmode=True):
        """
        :arg nodes: An array representing function values at each of the
                    dofs
        :arg firedrake_to_meshmode: *True* iff firedrake->meshmode, *False*
            if reordering meshmode->firedrake
        """
        # {{{ reorder data (Code adapted from
        # meshmode.mesh.processing.flip_simplex_element_group)

        # obtain function data in form [nelements][nunit_nodes]
        # and get flip mat
        # ( round to int bc applying on integers)
        flip_mat = np.rint(self.get_flip_matrix())
        if firedrake_to_meshmode:
            flip_mat = flip_mat.T
        cell_node_list = self.fd_function_space.cell_node_list
        data = nodes[cell_node_list]

        # flipping twice should be identity
        assert np.linalg.norm(
            np.dot(flip_mat, flip_mat)
            - np.eye(len(flip_mat))) < self._thresh

        # flip nodes that need to be flipped

        # if a vector function space, data array is shaped differently
        if self.fd_function_space.shape:
            data[self._orient < 0] = np.einsum(
                "ij,ejk->eik",
                flip_mat, data[self._orient < 0])
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
            # pytential wants [vector dims][nodes] not [nodes][vector dims]
            data = data.T.copy()
        else:
            data[self._orient < 0] = np.einsum(
                "ij,ej->ei",
                flip_mat, data[self._orient < 0])
            # convert from [element][unit_nodes] to
            # global node number
            data = data.flatten()

        # }}}

        return data

    def meshmode_mesh(self):
        if self._meshmode_mesh is None:

            order = self._ufl_base.degree()

            unit_nodes = self._get_unit_nodes()

            topological_dim = self._mesh_analog.topological_dimension()
            vertex_indices = self._mesh_analog._vertex_indices
            vertices = self._mesh_analog._vertices
            orient = self._mesh_analog._orient

            # {{{ Compute nodes

            ambient_dim = self._mesh_analog.geometric_dimension()
            nelements = vertex_indices.shape[0]
            nunit_nodes = unit_nodes.shape[1]
            unit_vertices = self._cell_analog._unit_vertices.T.copy()

            nodes = np.zeros((ambient_dim, nelements, nunit_nodes))
            for i, indices in enumerate(vertex_indices):
                elt_coords = np.zeros((ambient_dim, len(indices)))
                for j in range(elt_coords.shape[1]):
                    elt_coords[:, j] = vertices[:, indices[j]]
                A, b = _get_affine_mapping(unit_vertices, elt_coords)

                elt_nodes = np.matmul(A, unit_nodes) + b
                nodes[:, i, :] = elt_nodes[:, :]

            # }}}

            from meshmode.mesh import SimplexElementGroup
            group = SimplexElementGroup(order, vertex_indices, nodes,
                                        dim=topological_dim, unit_nodes=unit_nodes)

            from meshmode.mesh.processing import flip_simplex_element_group
            group = flip_simplex_element_group(vertices,
                                               group,
                                               orient < 0)

            groups = [group]

            facial_adj_grps = self._mesh_analog._facial_adjacency_groups
            boundary_tags = self._mesh_analog._boundary_tags

            from meshmode.mesh import Mesh
            self._meshmode_mesh = Mesh(vertices, groups,
                                       boundary_tags=boundary_tags,
                                       facial_adjacency_groups=facial_adj_grps)
            self._nodes = nodes

        return self._meshmode_mesh


class FiredrakeMeshmodeConverter:
    """
        Conversion :mod:`firedrake` to :mod:`meshmode`
    """
    def __init__(self, cl_ctx, dg_fspace_analog, bdy_id=None, **kwargs):

        degree = dg_fspace_analog._ufl_base.degree()
        fine_order = degree
        fmm_order = degree
        qbx_order = degree
        if 'fine_order' in kwargs:
            fine_order = kwargs['fine_order']
        if 'qbx_order' in kwargs:
            qbx_order = kwargs['qbx_order']
        if 'fmm_order' in kwargs:
            fmm_order = kwargs['fmm_order']

        factory = InterpolatoryQuadratureSimplexGroupFactory(degree)

        pre_density_discr = Discretization(
            cl_ctx,
            dg_fspace_analog.meshmode_mesh(),
            factory)

        self._domain_qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=fine_order,
            qbx_order=qbx_order,
            fmm_order=fmm_order)

        if bdy_id is not None:
            from meshmode.discretization.connection import \
                make_face_restriction
            self._domain_to_source = make_face_restriction(self._domain_qbx,
                                                           factory,
                                                           bdy_id)
            self._source_qbx = QBXLayerPotentialSource(
                self._domain_to_source.to_discr,
                fine_order=fine_order,
                qbx_order=qbx_order,
                fmm_order=fmm_order)
        else:
            self._domain_to_source = None
            self._source_qbx = self._domain_qbx

        self._dg_fspace_analog = dg_fspace_analog

    def convert(self, queue, weights):
        """
            Returns converted weights as an *np.array*

            Firedrake->meshmode conversion, converts to source mesh

            :arg queue: The pyopencl queue
                        NOTE: May pass *None* unless source is an interpolation
                              onto the boundary of the domain mesh
                        NOTE: Must be created from same cl_ctx this object
                              created with
            :arg weights:
                - An *np.array* with the weights representing the function or
                  discretization
                - a Firedrake :class:`Function`
        """
        if queue is None and self._domain_to_source is not None:
            raise ValueError("""Cannot pass *None* for :arg:`queue` if the
                              source mesh is not the whole domain""")

        # {{{ Convert data to np.array if necessary
        data = weights
        if isinstance(weights, fd.Function):
            if not self._dg_fspace_analog.is_valid(weights.function_space()):
                raise ValueError("Function not on valid function space for"
                                 " given this class's DGFunctionSpaceAnalog")
            data = weights.dat.data
        elif not isinstance(data, np.ndarray):
            raise ValueError("weights type not one of"
                " Firedrake.Function, np.array]")
        # }}}

        # Get the array with the re-ordering applied
        data = self._dg_fspace_analog.reorder_nodes(data)

        # {{{ if interpolation onto the source is required, do so
        if self._domain_to_source is not None:
            # if a vector function space, data is a np.array of arrays
            if self.fd_function_space.shape:
                data_array = []
                for arr in data:
                    data_array.append(
                        self._domain_to_source(queue,
                            cl.array.to_device(queue, arr)).get(queue=queue)
                    )
                data = np.array(data_array)
            else:
                data = cl.array.to_device(queue, data)
                data = \
                    self._domain_to_source(queue, data).\
                    with_queue(queue).get(queue=queue)
        # }}}

        return data


class MassConverter:
    def __init__(self):
        pass
