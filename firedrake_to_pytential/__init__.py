import pyopencl as cl

import numpy as np
from numpy import linalg as la

import firedrake as fd

from meshmode.mesh import MeshElementGroup
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind
from pytential.qbx import QBXLayerPotentialSource
from pytential.target import PointsTarget


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
        Must be a DG order 1 family with triangle elements with
        topological dimension 2.
    :arg ambient_dim: Must be at least the topological dimension.
    """

    # assert that Function Space is using DG elements of
    # appropriate type
    if function_space.ufl_element().family() != 'Discontinuous Lagrange':
        raise TypeError("""Function space must use Discontinuous
                         Lagrange elements""")

    if function_space.mesh().topological_dimension() != 2:
        raise TypeError("""Only function spaces with meshes of
                        topological dimension 2 are supported""")

    if function_space.finat_element.degree != 1:
        raise TypeError("""Only function spaces with elements of
                        degree 1 are supported""")

    if str(function_space.ufl_cell()) != 'triangle':
        raise TypeError("Only triangle reference elements are supported")

    # assert ambient dimension is big enough
    if ambient_dim < function_space.mesh().topological_dimension():
        raise ValueError("""Desired ambient dimension of meshmode Mesh must be at
                            least the topological dimension of the firedrake
                            FunctionSpace""")

    mesh = function_space.mesh()
    mesh.init()
    if ambient_dim is None:
        ambient_dim = mesh.geometric_dimension()
    dim = mesh.topological_dimension()

    # {{{ Construct a SimplexElementGroup
    # Note: not using meshmode.mesh.generate_group
    #       because I need to keep the node ordering
    order = function_space.ufl_element().degree()

    # FIXME: We may want to allow variable dtype
    coords = np.array(mesh.coordinates.dat.data, dtype=np.float64)
    coords_fn_space = mesh.coordinates.function_space()

    # <from meshmode docs:>
    # An array of (nelements, ref_element.nvertices)
    # of (mesh-wide) vertex indices
    vertex_indices = np.copy(coords_fn_space.cell_node_list)

    # get (mesh-wide) vertex coordinates,
    # pytential wants [ambient_dim][nvertices]
    vertices = np.array(coords)
    vertices = vertices.T.copy()
    vertices.resize((ambient_dim,)+vertices.shape[1:])

    # FIXME: Node construction ONLY works for order 1
    #        elements
    #
    # construct the nodes
    # <from meshmode docs:> an array of node coordinates
    # (mesh.ambient_dim, nelements, nunit_nodes)

    node_coordinates = coords
    # give nodes as [nelements][nunit_nodes][dim]
    nodes = [[node_coordinates[inode] for inode in indices]
             for indices in vertex_indices]
    nodes = np.array(nodes)

    # convert to [ambient_dim][nelements][nunit_nodes]
    nodes = np.transpose(nodes, (2, 0, 1))
    if(nodes.shape[0] < ambient_dim):
        nodes = np.resize(nodes, (ambient_dim,) + nodes.shape[1:])
        nodes[-1, :, :] = 0

    # FIXME : This only works for firedrake mesh with
    #         geometric dimension (in meshmode language--ambient dim)
    #         of 2.
    #         (NOTE: for meshes embedded in 3-space,
    #                try using MeshGeoemtry.init_cell_orientations?)
    #
    # Now we ensure that the vertex indices
    # give a positive ordering when
    # SimplexElementGroup.face_vertex_indices is called

    #from meshmode.mesh.io import from_vertices_and_simplices
    #group = from_vertices_and_simplices(vertices, vertex_indices, 1, True)
    if mesh.geometric_dimension() == 2:
        """
        This code is nearly identical to
        meshmode.mesh.io.from_vertices_and_simplices.
        The reason we don't simply call the function is that
        for higher orders (which we hope to eventually use)
        we need to use Firedrake's nodes
        """
        if ambient_dim == 2:
            from meshmode.mesh.generation import make_group_from_vertices

            order_one_grp = make_group_from_vertices(vertices,
                                                     vertex_indices, 1)

            from meshmode.mesh.processing import (
                    find_volume_mesh_element_group_orientation,
                    flip_simplex_element_group)
            # This function fails in ambient_dim 3, hence the
            # separation between ambient_dim 2 and 3
            orient = \
                find_volume_mesh_element_group_orientation(vertices,
                                                           order_one_grp)
        else:
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
    else:
        raise TypeError("Only geometric dimension 2 is supported")

    from meshmode.mesh import SimplexElementGroup
    group = SimplexElementGroup(order, vertex_indices, nodes, dim=2)

    from meshmode.mesh.processing import flip_simplex_element_group
    group = flip_simplex_element_group(vertices,
                                       group,
                                       orient < 0)

    # FIXME : only supports one kind of element, is this OK?
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
        connectivity[(dim - 1, 0)]  # maps faces to vertices

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
        :class:`Discretization', then converts functions back and forth between
        the two

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

        .. attribute:: target_points

            A *pytential.target* :class:`PointsTarget` of the target. Either
            all of the nodes, or the boundary passed in.

        .. attribute:: source_is_domain

            Boolean value, true if source mesh is same as domain

        .. attribute:: target_is_domain

            Boolean value, true if :attr:`target_points` covers the whole
            domain mesh.

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

        .. attribute:: _target_embedding

            An array, it should be that *_target_embedding[i]* represents
            the node index in *mesh_map['domain']* of the *i*th
            entry in :attr:`target_points`.
    """

    def __init__(self, cl_ctx, queue, function_space, ambient_dim=None,
                 source_bdy_id=None, target_bdy_id=None, thresh=1e-13):
        self._thresh = thresh
        """
        :arg cl_ctx: A pyopencl context
        :arg queue: A pyopencl command queue created on :arg:`cl_ctx`
        :arg function_space: Sets :attr:`fd_function_space`
        :arg ambient_dim: Sets :attr:`ambient_dim`. If none, this defaults to
                          *fd_function_space.geometric_dimension()*
        :arg source_bdy_id: See :attr:`mesh_map`. If *None*,
                            source defaults to target.
        :arg target_bdy_id: See :attr:`target_points`. If *None*,
                            target defaults to the whole domain.

        :arg thresh: Used as threshold for some float equality checks
                     (specifically, those involved in asserting
                     positive ordering of faces)
        """

        # {{{ Declare attributes

        self.fd_function_space = function_space
        self.mesh_map = {'domain': None, 'source': None}
        self.qbx_map = {'domain': None, 'source': None}
        self.target_points = None
        self.source_is_domain = (source_bdy_id is None)
        self.target_is_domain = (target_bdy_id is None)
        self.ambient_dim = ambient_dim
        self._domain_to_source = None
        self._orient = None
        self._flip_matrix = None
        self._target_embedding = None

        # }}}

        # {{{ construct domain meshmode mesh and qbx

        # If ambient dim is unset, set to geometric dimension ofd
        # fd function space
        if self.ambient_dim is None:
            self.ambient_dim = self.fd_function_space.mesh().geometric_dimension()

        # Ensure co-dimension is 0 or 1
        codimension = self.ambient_dim - function_space.mesh().topological_dimension()
        if codimension not in [0, 1]:
            raise ValueError('Co-dimension is %s, should be 0 or 1' % (codimension))

        # create mesh and qbx
        self.mesh_map['domain'], self._orient = \
            _convert_function_space_to_meshmode(self.fd_function_space,
                                                self.ambient_dim)
        pre_density_discr = Discretization(
            cl_ctx,
            self.mesh_map['domain'],
            InterpolatoryQuadratureSimplexGroupFactory(1))

        # FIXME : Assumes order 1
        # FIXME : Do I have the right thing for the various orders?
        self.qbx_map['domain'] = QBXLayerPotentialSource(pre_density_discr,
                                            fine_order=1,
                                            qbx_order=1,
                                            fmm_order=1)
        # }}}

        # {{{ Perform boundary interpolation if required
        if self.source_is_domain:
            self.mesh_map['source'] = self.mesh_map['domain']
            self.qbx_map['source'] = self.qbx_map['domain']
        else:
            from meshmode.discretization.connection import \
                make_face_restriction

            # FIXME : Assumes order 1
            self._domain_to_source = make_face_restriction(
                self.qbx_map['domain'].density_discr,
                InterpolatoryQuadratureSimplexGroupFactory(1),
                source_bdy_id)

            self.mesh_map['source'] = self._domain_to_source.to_discr.mesh
            self.qbx_map['source'] = QBXLayerPotentialSource(
                self._domain_to_source.to_discr,
                fine_order=1,
                qbx_order=1,
                fmm_order=1)

        # }}}

        # {{{ Compute target_points

        # If *target_bdy_id* is *None*, then target points
        # is the whole domain
        domain_discr = self.qbx_map['domain'].density_discr
        nodes = domain_discr.nodes().with_queue(queue).get(queue=queue)
        nnodes = nodes.shape[1]
        if target_bdy_id is None:
            self._target_embedding = np.arange(nnodes)
        else:
            if target_bdy_id not in self.mesh_map['domain'].btag_to_index:
                raise ValueError("target_bdy_id is invalid")
            btag_index = self.mesh_map['domain'].btag_to_index[target_bdy_id]
            btag_mask = (1 << btag_index)

            bdy_fagrp = self.mesh_map['domain'].facial_adjacency_groups[0].get(None, None)
            if bdy_fagrp is None:
                raise ValueError("No boundary found on domain\n")
            # list of (element, face number) of faces
            # that lie on the target boundary
            elems_and_faces_on_target = []
            for i, nbr in enumerate(bdy_fagrp.neighbors):
                assert nbr < 0
                btags = -nbr
                if (btags & btag_mask):
                    element = bdy_fagrp.elements[i]
                    face = bdy_fagrp.element_faces[i]
                    elems_and_faces_on_target.append((element, face))

            # See which nodes are "on" each face
            grp = self.mesh_map['domain'].groups[0]
            unit_nodes = grp.unit_nodes
            face_vertex_indices = grp.face_vertex_indices()
            vertex_unit_coordinates = grp.vertex_unit_coordinates()
            unit_nodes_on_face = []
            for face in face_vertex_indices:
                unit_nodes_on_face.append([])

                face_array = np.array(face)
                # (pts on face, dim)
                coords = vertex_unit_coordinates[face_array]
                # shape (dim, nspanning vects)
                spanning_vects = (coords[1:] - coords[0]).T

                # *un* has shape (dim)
                for i, un in enumerate(unit_nodes.T):
                    # if *un* is in the span of coords, is on face
                    vect = un - coords[0]
                    _, residual, _, _ = np.linalg.lstsq(spanning_vects, vect, rcond=None)
                    if np.linalg.norm(residual) < self._thresh:
                        unit_nodes_on_face[-1].append(i)

            # Create target embedding
            nunit_nodes = grp.nunit_nodes
            self._target_embedding = set()
            for element, face in elems_and_faces_on_target:
                node_nr_elem_base = grp.node_nr_base + element * nunit_nodes
                for iunit_node in unit_nodes_on_face[face]:
                    node_nr = node_nr_elem_base + iunit_node
                    self._target_embedding.add(node_nr)
            self._target_embedding = np.array(list(self._target_embedding), dtype=np.int32)

        if self._target_embedding.shape[0] == 0:
            raise UserWarning("No nodes on boundary with id %s found" % (target_bdy_id))

        self.target_points = cl.array.to_device(queue, nodes[:, self._target_embedding])
        self.target_points = PointsTarget(self.target_points)
        # }}}

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

    def _reorder_nodes(self, nodes, invert=False):
        """
        :arg nodes: An array representing function values at each of the
                    dofs
        :arg invert: False if and only if converting firedrake to
                      meshmode ordering, else does meshmode to firedrake
                      ordering
        """
        if self._flip_matrix is None:
            self._compute_flip_matrix()

        # {{{ reorder data (Code adapted from
        # meshmode.mesh.processing.flip_simplex_element_group)

        # obtain function data in form [nelements][nunit_nodes]
        # and get flip mat
        # FIXME
        flip_mat = np.rint(self._flip_matrix)
        if not invert:
            cell_node_list = self.fd_function_space.cell_node_list
            data = nodes[cell_node_list]
        else:
            nelements = self.mesh_map['domain'].groups[0].nelements
            nunit_nodes = self.mesh_map['domain'].groups[0].nunit_nodes
            data = np.reshape(nodes, (nelements, nunit_nodes))
            flip_mat = flip_mat.T

        # flipping twice should be identity
        assert np.linalg.norm(
            np.dot(flip_mat, flip_mat)
            - np.eye(len(flip_mat))) < self._thresh

        # flip nodes that need to be flipped

        data[self._orient < 0] = np.einsum(
            "ij,ej->ei",
            flip_mat, data[self._orient < 0])

        # }}}

        # convert from [element][unit_nodes] to
        # global node number
        data = data.flatten()

        return data

    def __call__(self, queue, weights, invert=False):
        """
            Returns converted weights as an *np.array*

            Firedrake->meshmode we interpolate onto the source mesh.
            meshmode->Firedrake we interpolate from the target mesh.

            :arg queue: The pyopencl queue
                        NOTE: May pass *None* unless source is an interpolation
                              onto the boundary of the domain mesh
            :arg weights: One of
                - An *np.array* with the weights representing the function or
                  discretization
                - a Firedrake :class:`Function`
                - a pyopencl.array.Array representing the discretization
            :arg invert: True iff meshmode to firedrake instead of
                         firedrake to meshmode
        """
        if queue is None and (invert is False and not self.source_is_domain):
            raise ValueError("""When converting from firedrake to meshmode mesh,
                              cannot pass *None* for :arg:`queue` if the
                              source mesh is not the whole domain""")

        # {{{ Convert data to np.array if necessary
        data = weights
        if isinstance(weights, fd.Function):
            assert (not invert)
            data = weights.dat.data
        elif isinstance(weights, cl.array.Array):
            assert invert
            data = weights.get(queue=queue)
        elif not isinstance(data, np.ndarray):
            raise ValueError("""weights type not one of [Firedrake.Function,
                             pyopencl.array.Array, np.array]""")
        # }}}

        # If inverting (i.e. meshmode->Firedrake),
        # and the target mesh is an embedding,
        # undo the interpolation and store in *data*.
        if invert and not self.target_is_domain:
            # {{{ Invert the interpolation
            arr = np.zeros(self.mesh_map['domain'].groups[0].nnodes,
                           dtype=data.dtype)
            for index, weight in zip(self._target_embedding, data):
                arr[index] = weight
            data = arr
            # }}}

        # Get the array with the re-ordering applied
        data = self._reorder_nodes(data, invert)

        # {{{ if interpolation onto the source is required (while converting
        #     Firedrake->meshmode), do so
        if not invert and not self.source_is_domain:
            data = cl.array.to_device(queue, data)
            data = \
                self._meshmode_connections['source'](queue, data).\
                with_queue(queue).get(queue=queue)
        # }}}

        return data
