import pyopencl as cl
import six

import numpy as np
from numpy import linalg as la

import firedrake as fd

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential.qbx import QBXLayerPotentialSource
from warnings import warn


def _get_unit_nodes(function_space):
    order = function_space.ufl_element().degree()
    fe = function_space.finat_element
    cell = fe.cell
    node_nr_to_coords = {}

    for dim, element_nrs in six.iteritems(fe.entity_support_dofs()):
        for element_nr, node_list in six.iteritems(element_nrs):
            pts_on_element = cell.make_points(dim, element_nr, order)
            i = 0
            for node_nr in node_list:
                if node_nr not in node_nr_to_coords:
                    node_nr_to_coords[node_nr] = pts_on_element[i]
                    i += 1

    # (nunit_nodes, dim)
    unit_nodes = [node_nr_to_coords[i] for i in range(len(node_nr_to_coords))]
    unit_nodes = np.array(unit_nodes)
    # (dim, nunit_nodes)
    unit_nodes = np.copy(unit_nodes.T)

    fd_verts = cell.get_vertices()
    fd_verts = np.copy(np.array(fd_verts).T)
    if cell.get_dimension() == 1:
        unit_coords = [(-1,), (1,)]
    elif cell.get_dimension() == 2:
        # from modepy
        unit_coords = [(-1, -1), (1, -1), (-1, 1)]
    elif cell.get_dimension() == 3:
        # from modepy
        unit_coords = [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    else:
        raise ValueError("Only dimension 1, 2 and 3 cells supported, provided cell"
            "has dimension %s" % (cell.get_dimension()))
    unit_coords = np.copy(np.array(unit_coords).T)

    # A * fd_verts + b -> unit_coords
    fd_span_vects = fd_verts[:, 1:] - fd_verts[:, 0, np.newaxis]
    unit_coord_span_vects = unit_coords[:, 1:] - unit_coords[:, 0, np.newaxis]
    A = np.linalg.solve(fd_span_vects, unit_coord_span_vects)
    b = np.matmul(A, -fd_verts[:, 0]) + unit_coords[:, 0]

    new_unit_nodes = np.zeros(unit_nodes.shape)
    for i in range(unit_nodes.shape[1]):
        new_unit_nodes[:, i] = np.matmul(A, unit_nodes[:, i]) + b
    return new_unit_nodes


def _convert_function_space_to_meshmode(function_space, ambient_dim):
    """
    This converts a :class:`FunctionSpace` to a meshmode :class:`Mesh`
    with the given ambient dimension. Creates a 1-1 correspondence of

    firedrake dofs <-> meshmode nodes

    firedrake vertices <-> meshmode vertices

    firedrake faces <-> meshmode faces

    etc. Note that the 1-1 correspondence may be, in general, a non-trivial
    re-ordering.

    The resultant :class:`Mesh` will have one group of type
    :class:`SimplexElementGroup`.

    Returns a tuple (:class:`Mesh`, *np.array*). The first entry
    is converted :class:`FunctionSpace`, the second entry *b* is an
    array of floats representing orientations. For the *i*th
    element in the mesh's *.groups[0]* group, its *i*th (group-local)
    element had a positive orientation in :arg:`function_space`
    if *b[i] >= 0* and a negative orientation if *b[i] < 0*.

    :arg function_space: A Firedrake :class:`FunctionSpace`.
        Must be a DG family with triangle elements. The geometric
        and topological dimension of the mesh must be
        either 2 or 3 (i.e. (2, 2), (2, 3), or (3, 3))

        NOTE: If the geometric dimension is 3 and topological
        dimension is 2 (i.e. a 2-surface embedded in 3-space),
        then the function :function:`init_cell_orientations`
        MUST have been called on the underlying mesh of :arg:`function_space`.

        NOTE: If one wishes to have a mesh which in firedrake
        has geometric dimension 2 reside in 3-space when converted
        to meshmode, use :arg:`ambient_dim` to do this.

    :arg ambient_dim: Must be at least the geometric dimension.
        Co-dimension of mesh must be 1 or 0.
    """

    # assert that Function Space is using DG elements of
    # appropriate type
    if function_space.ufl_element().family() != 'Discontinuous Lagrange':
        raise TypeError("Function space must use Discontinuous "
                         "Lagrange elements, you are trying to use %s "
                         "elements" % (function_space.ufl_element().family()))

    topological_dim = function_space.mesh().topological_dimension()
    geometric_dim = function_space.mesh().geometric_dimension()

    #if function_space.mesh().topological_dimension() != 2:
    #    raise TypeError("""Only function spaces with meshes of
    #                    topological dimension 2 are supported""")
    if topological_dim not in [1, 2, 3]:
        raise ValueError("Only topological dimension in [1, 2, 3] is supported, "
        "this mesh has topological dimension %s." % (topological_dim))
    if geometric_dim not in [1, 2, 3]:
        raise ValueError("Only geometric dimension in [1, 2, 3] is supported, "
        "this mesh has geometric dimension %s." % (geometric_dim))

    if str(function_space.ufl_cell()) not in ['triangle', 'triangle3D',
            'tetrahedron', 'interval', 'interval2D']:
        raise TypeError("Only triangle and tetrahedron"
            "reference elements are supported. "
            "The given function space uses %s elements" %
            (function_space.ufl_cell()))

    # assert ambient dimension is not too big
    if ambient_dim < geometric_dim:
        raise ValueError("Desired ambient dimension (in this case, ambient dim=%s) "
        "of the meshmode Mesh must be at "
        "least the geometric dimension of the firedrake "
        "FunctionSpace (in this case, geometric dim=%s) " %
        (ambient_dim, geometric_dim))

    # Ensure co-dimension is 0 or 1
    if ambient_dim is not None:
        codimension = ambient_dim - topological_dim
        if codimension not in [0, 1]:
            raise ValueError('Co-dimension is %s, should be 0 or 1' % (codimension))

    mesh = function_space.mesh()
    mesh.init()
    if ambient_dim is None:
        ambient_dim = geometric_dim

    # {{{ Construct a SimplexElementGroup
    # Note: not using meshmode.mesh.generate_group
    #       because I need to keep the node ordering
    order = function_space.ufl_element().degree()
    unit_nodes = _get_unit_nodes(function_space)

    # TODO: We may want to allow variable dtype?
    coords = np.array(mesh.coordinates.dat.data).real
    coords_fn_space = mesh.coordinates.function_space()

    # <from meshmode docs:>
    # An array of (nelements, ref_element.nvertices)
    # of (mesh-wide) vertex indices
    vertex_indices = np.copy(coords_fn_space.cell_node_list)

    # get (mesh-wide) vertex coordinates,
    # pytential wants [ambient_dim][nvertices], but for now we
    # write it as [geometric dim][nvertices]
    vertices = np.array(coords)
    vertices = vertices.T.copy()

    # construct the nodes
    # <from meshmode docs:> an array of node coordinates
    # (mesh.ambient_dim, nelements, nunit_nodes)

    vector_fspace = fd.VectorFunctionSpace(mesh,
                                           function_space.ufl_element().family(),
                                           degree=order,
                                           dim=geometric_dim)
    xx = fd.SpatialCoordinate(mesh)
    node_coordinates = fd.Function(vector_fspace).interpolate(xx).dat.data
    # give nodes as [nelements][nunit_nodes][ambient_dim]
    nodes = [[node_coordinates[inode] for inode in indices]
             for indices in function_space.cell_node_list]
    nodes = np.array(nodes).real
    # eventually convert to [ambient_dim][nunit_nodes][nelements]
    # (but for now [geometric dim][nunit_nodes][nelements])
    nodes = np.transpose(nodes, (2, 0, 1))

    # Now we ensure that the vertex indices
    # give a positive ordering when
    # SimplexElementGroup.face_vertex_indices is called
    orient = None
    if ambient_dim == topological_dim:
        # In this case we have either a 2-surface in 2-space
        # or a 3-surface in 3-space
        """
        This code is nearly identical to
        meshmode.mesh.io.from_vertices_and_simplices.
        The reason we don't simply call the function is that
        for higher orders we need to use Firedrake's nodes
        """
        from meshmode.mesh.generation import make_group_from_vertices

        order_one_grp = make_group_from_vertices(vertices, vertex_indices, 1)

        from meshmode.mesh.processing import \
            find_volume_mesh_element_group_orientation
        # This function fails in ambient_dim 3, hence the
        # separation between ambient_dim 2 and 3
        orient = \
            find_volume_mesh_element_group_orientation(vertices,
                                                       order_one_grp)
    else:
        # Resize if ambient_dim > geometric_dim
        """
        One note: we have to zero out the added dimension in nodes because
        it is not  single-segment array--so cannot be resized in place.
        The *np.resize* function copies data from the original array
        into the added dimensions of the new one

        For vertices on the other hand, it is a single-segment array,
        so when it is resized the added entries are set to 0.
        """
        if ambient_dim > geometric_dim:
            nodes = np.resize(nodes, (ambient_dim,)+nodes.shape[1:])
            nodes[-1, :, :] = 0.0
            vertices.resize((ambient_dim,)+vertices.shape[1:])

        # Now get the orientations
        if topological_dim == 1 and geometric_dim == 1:
            # In this case we have a 1-dimensional mesh being
            # embedded into 2-space
            orient = np.zeros(vertex_indices.shape[0])
            for i, simplex in enumerate(vertex_indices):
                va = vertices[simplex[0]]
                vb = vertices[simplex[1]]
                orient[i] = vb - va
        if topological_dim == 1 and geometric_dim == 2:
            # In this case we have a 1-surface embedded in 2-space
            # FIXME : Here, we just assume all are positively oriented
            orient = np.ones(vertex_indices.shape[0])
        if topological_dim == 2 and geometric_dim == 2:
            # In this case we have a 2-dimensional mesh being
            # embedded into 3-space

            orient = np.zeros(vertex_indices.shape[0])
            # Just use cross product to tell orientation
            for i, simplex in enumerate(vertex_indices):
                #       v_from
                #        o._
                #       /   ~._
                #      /       ~._
                #     /           ~._
                # va o---------------o vb
                v_from = vertices[:, simplex[0]]
                va = vertices[:, simplex[1]]
                vb = vertices[:, simplex[2]]
                orient[i] = np.cross(va - v_from, vb - v_from)[-1]
        elif topological_dim == 2 and geometric_dim == 3:
            # In this case we have a 2-surface embedded in 3-space

            orient = function_space.mesh().cell_orientations().\
                dat.data.astype(np.float64)
            # Convert (0 ==> negative, 1 ==> positive) to
            # (-1 ==> negative, 1 ==> positive)
            orient *= 2
            orient -= np.ones(orient.shape)

    # Make sure the mesh fell into one of the above cases
    assert orient is not None

    from meshmode.mesh import SimplexElementGroup
    group = SimplexElementGroup(order, vertex_indices, nodes, dim=topological_dim,
                                unit_nodes=unit_nodes)

    from meshmode.mesh.processing import flip_simplex_element_group
    group = flip_simplex_element_group(vertices,
                                       group,
                                       orient < 0)

    # TODO: only supports one kind of element, is this OK?
    groups = [group]

    from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
    boundary_tags = [BTAG_ALL, BTAG_REALLY_ALL]

    efacets = mesh.exterior_facets
    if efacets.unique_markers is not None:
        for tag in efacets.unique_markers:
            boundary_tags.append(tag)
    boundary_tags = tuple(boundary_tags)

    # fvi_to_tags maps frozenset(vertex indices) to tags
    fvi_to_tags = {}
    connectivity = function_space.finat_element.cell.\
        connectivity[(topological_dim - 1, 0)]  # maps faces to local vertex indices

    original_vertex_indices_ordering = np.array(
        coords_fn_space.cell_node_list)
    for i, (icell, ifac) in enumerate(zip(
            efacets.facet_cell, efacets.local_facet_number)):
        # unpack arguments
        ifac, = ifac
        icell, = icell
        # record face vertex indices to tag map
        facet_indices = connectivity[ifac]
        fvi = frozenset(original_vertex_indices_ordering[icell]
                        [list(facet_indices)])
        fvi_to_tags.setdefault(fvi, [])
        fvi_to_tags[fvi].append(efacets.markers[i])

    from meshmode.mesh import _compute_facial_adjacency_from_vertices
    facial_adj_grps = _compute_facial_adjacency_from_vertices(
        groups,
        boundary_tags,
        np.int32, np.int8,
        face_vertex_indices_to_tags=fvi_to_tags)

    from meshmode.mesh import Mesh
    return (Mesh(vertices, groups,
                 boundary_tags=boundary_tags,
                 facial_adjacency_groups=facial_adj_grps),
            orient)


class FiredrakeMeshmodeConnection:
    """
        This class takes a firedrake :class:`FunctionSpace`, makes a meshmode
        :class:`Discretization', then converts functions from Firedrake
        to meshmode

        .. attribute:: fd_function_space

            The firedrake :class:`FunctionSpace` object. It must meet
            the requirements as described by :arg:`function_space` in the
            function *_convert_function_space_to_meshmode*

        .. attribute:: mesh_map

            A :class:`dict` mapping certain keywords to a meshmode
            :class:`mesh`. Keywords are as follows:

            The keyword *'domain'*. This maps to the
            meshmode :class:`Mesh` object associated to :attr:`fd_function_space`.
            For more details, see the function *_convert_function_space_to_meshmode*.

            The keyword *'source'*. This is the default
            mesh for Firedrake->meshmode conversion of functions. This mesh
            is either the same as *mesh_map['domain']* or the result
            of *meshmode.discretization.connection.make_face_restriction*
            being called on *mesh_map['domain']* with a boundary tag given
            at construction.

            No other keywords are allowed. The above keywords are always present.

        .. attribute:: qbx_map

            A :class:`dict` mapping certain keywords to a *pytential.qbx*
            :class:`QBXLayerPotentialSource`. Keywords are only
            *'domain'* and *'source'*

            A keyword *'w'* maps to a :class:`QBXLayerPotentialSource`
            object created on the mesh *mesh_map['w']*.

            For now, all are created with order 1.

        .. attribute:: source_is_domain

            Boolean value, true if source mesh is same as domain

        .. attribute:: ambient_dim

            The ambient dimension of the meshmode mesh which corresponds
            to :attr:`fd_function_space`. Note that to compute layer potentials,
            the co-dimension of the mesh must be 1.

            Example: If one is going to compute 3-dimensional layer potentials
                     on a 2-dimensional mesh, one would want the mesh to
                     have ambient dimension 3.

            Example: If one is going to compute 2-dimensional layer potentials
                     on the boundary of a 2-dimensional mesh, one would make
                     the mesh with ambient dimension 2, since the boundary
                     would then have codimension 1.

            We enforce that the mesh associated to :attr:`fd_function_space`
            would have co-dimension 0 or 1 when embedded in a space of dimension
            :attr:`ambient_dim`. (i.e.
            ``ambient_dim - fd_function_space.topological_dimension() = 0, 1``).

        .. attribute:: _domain_to_source

             If :attr:`source_is_domain` is true, this is *None*. Else, this
             is a :class:`DiscretizationConnection` from *mesh_map['domain']*
             to *mesh_map['source']*.

        .. attribute:: _orient

            An *np.array* of shape *(meshmode_mesh.groups[0].nelements)* with
            positive entries for each positively oriented element,
            and a negative entry for each negatively oriented element.

        .. attribute:: _flip_matrix

            Used to re-orient elements. For more details, look at the
            function *meshmode.mesh.processing.flip_simplex_element_group*.
    """

    def __init__(self, cl_ctx, function_space, ambient_dim=None,
                 source_bdy_id=None, **kwargs):
        """
        :arg cl_ctx: A pyopencl context
        :arg function_space: Sets :attr:`fd_function_space`
        :arg ambient_dim: Sets :attr:`ambient_dim`. If none, this defaults to
                          *fd_function_space.geometric_dimension()*
        :arg source_bdy_id: See :attr:`mesh_map`. If *None*,
                            source defaults to domain.

        :kwarg thresh: Used as threshold for some float equality checks
                     (specifically, those involved in asserting
                     positive ordering of faces). Defaults to
                     *1e-10*.
        :kwarg with_refinement: If *True*, the :attr:`qbx_map`
            at *'source'* will be called 'with_refinement' (see
            :mod:`pytential`). Defaults to *False*.
        :kwarg fine_order: See :mod:`pytential`, used for
            :class:`QBXLayerSourcePotential` construction. Defaults to
            degree of elements used in *function_space*
        :kwarg qbx_order: See above
        :kwarg fmm_order: See above
        """
        # {{{ Handle kwargs

        expected_kwargs = set(['thresh', 'with_refinement', 'fine_order',
            'qbx_order', 'fmm_order'])
        # make sure not receiving extraneous kwargs
        if set(kwargs.keys()) - expected_kwargs:
            raise ValueError("Received unexpected keyword argument(s) %s" %
                (set(kwargs.keys()) - expected_kwargs))

        self._thresh = 1e-10
        if 'thresh' in kwargs:
            self._thresh = kwargs['thresh']

        with_refinement = False
        if 'with_refinement' in kwargs:
            with_refinement = kwargs['with_refinement']

        # Degree of function space
        degree = function_space.ufl_element().degree()

        fine_order = degree
        if 'fine_order' in kwargs:
            fine_order = kwargs['fine_order']
        qbx_order = degree
        if 'qbx_order' in kwargs:
            qbx_order = kwargs['qbx_order']
        fmm_order = degree
        if 'fmm_order' in kwargs:
            fmm_order = kwargs['fmm_order']
        # }}}

        # {{{ Declare attributes

        self.fd_function_space = function_space
        self.mesh_map = {'domain': None, 'source': None}
        self.qbx_map = {'domain': None, 'source': None}
        self.source_is_domain = (source_bdy_id is None)
        self.ambient_dim = ambient_dim
        self._domain_to_source = None
        self._orient = None
        self._flip_matrix = None

        # }}}

        # {{{ construct domain meshmode mesh and qbx

        # If ambient dim is unset, set to geometric dimension ofd
        # fd function space
        if self.ambient_dim is None:
            self.ambient_dim = self.fd_function_space.mesh().geometric_dimension()

        # create mesh and qbx
        self.mesh_map['domain'], self._orient = \
            _convert_function_space_to_meshmode(self.fd_function_space,
                                                self.ambient_dim)
        pre_density_discr = Discretization(
            cl_ctx,
            self.mesh_map['domain'],
            InterpolatoryQuadratureSimplexGroupFactory(degree))

        # FIXME : Do I have the right thing for the various orders?
        self.qbx_map['domain'] = QBXLayerPotentialSource(pre_density_discr,
                                            fine_order=fine_order,
                                            qbx_order=qbx_order,
                                            fmm_order=fmm_order)
        # }}}

        # {{{ Perform boundary interpolation if required
        if self.source_is_domain:
            self.mesh_map['source'] = self.mesh_map['domain']
            self.qbx_map['source'] = self.qbx_map['domain']
        else:
            from meshmode.discretization.connection import \
                make_face_restriction

            self._domain_to_source = make_face_restriction(
                self.qbx_map['domain'].density_discr,
                InterpolatoryQuadratureSimplexGroupFactory(degree),
                source_bdy_id)

            # FIXME : Do I have the right thing for the various orders?
            self.mesh_map['source'] = self._domain_to_source.to_discr.mesh
            self.qbx_map['source'] = QBXLayerPotentialSource(
                self._domain_to_source.to_discr,
                fine_order=fine_order,
                qbx_order=qbx_order,
                fmm_order=fmm_order)

        # }}}

        if with_refinement:
            self.qbx_map['source'], _ = self.qbx_map['source'].with_refinement()

    def _compute_flip_matrix(self):
        #This code adapted from *meshmode.mesh.processing.flip_simplex_element_group*

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

    def _reorder_nodes(self, nodes, invert):
        """
        :arg nodes: An array representing function values at each of the
                    dofs
        :arg invert: True iff firedrake->meshmode, else False
        """
        if self._flip_matrix is None:
            self._compute_flip_matrix()

        # {{{ reorder data (Code adapted from
        # meshmode.mesh.processing.flip_simplex_element_group)

        # obtain function data in form [nelements][nunit_nodes]
        # and get flip mat
        # ( round to int bc applying on integers)
        flip_mat = np.rint(self._flip_matrix)
        if invert:
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

    def __call__(self, queue, weights, invert=False):
        """
            Returns converted weights as an *np.array*

            Firedrake->meshmode we interpolate onto the source mesh.
            WARNING:
            meshmode->Firedrake we interpolate ONLY from the domain mesh
            *there is NO check for this*

            :arg queue: The pyopencl queue
                        NOTE: May pass *None* unless source is an interpolation
                              onto the boundary of the domain mesh
                        NOTE: Must be created from same cl_ctx this object
                              created with
            :arg weights:
                - An *np.array* with the weights representing the function or
                  discretization
                - a Firedrake :class:`Function`

            :arg invert: *True* iff going meshmode->Firedrake
        """
        if queue is None and not self.source_is_domain:
            raise ValueError("""When converting from firedrake to meshmode mesh,
                              cannot pass *None* for :arg:`queue` if the
                              source mesh is not the whole domain""")

        # {{{ Convert data to np.array if necessary
        data = weights
        if isinstance(weights, fd.Function):
            data = weights.dat.data
        elif not isinstance(data, np.ndarray):
            raise ValueError("weights type not one of"
                " Firedrake.Function, np.array]")
        # }}}

        # Get the array with the re-ordering applied
        data = self._reorder_nodes(data, invert)

        # {{{ if interpolation onto the source is required, do so
        if not invert and not self.source_is_domain:
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

            The :mod:`firedrake` :class:`mesh` this object is an analog to
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

        # Reorder vertex indices
        self._vertex_indices = group.vertex_indices

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

        original_vertex_indices_ordering = np.array(
            vertices_fn_space.cell_node_list)
        for i, (icell, ifac) in enumerate(zip(
                efacets.facet_cell, efacets.local_facet_number)):
            # unpack arguments
            ifac, = ifac
            icell, = icell
            # record face vertex indices to tag map
            facet_indices = connectivity[ifac]
            fvi = frozenset(original_vertex_indices_ordering[icell]
                            [list(facet_indices)])
            fvi_to_tags.setdefault(fvi, [])
            fvi_to_tags[fvi].append(efacets.markers[i])

        from meshmode.mesh import _compute_facial_adjacency_from_vertices
        """
            NOTE : This relies HEAVILY on the fact that elements are *not*
                   reordered at any time, and that *_vertex_indices*
                   are not reordered anymore.
        """
        self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
            groups,
            boundary_tags,
            np.int32, np.int8,
            face_vertex_indices_to_tags=fvi_to_tags)
        # }}}

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


# TODO : Instead of a FunctionSpaceAnalog, maybe make
# a Super-Mesh analog, since really ONLY need the reference
# element. i.e. Figure out the least you need.
class FunctionSpaceAnalog:
    """
        This takes a :mod:`firedrake` :class:`MeshGeometry`
        and converts its data into :mod:`meshmode` format.

        .. attribute ::  _mesh_analog

            A :class:`MeshAnalog` corresponding to the given
            function space's mesh

        .. attribute :: _function_space

            The :class:`FunctionSpace` this is an analog of.
            We require *"DG"* elements.
    """

    def __init__(self, function_space, mesh_analog):

        # {{{ Compute reference element
        # }}}

        # {{{ Compute nodes
        # }}}

        # {{{ Compute flip-matrix and re-order nodes
        # }}}

        # {{{ Create :mod:`mehshmode` :class:`Mesh`
        # }}}

        self._mesh_analog = mesh_analog
        self._function_space = function_space

    def _get_flip_matrix(self):
        pass
