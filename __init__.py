"""
Idea for mesh interaction:
    Only deal with 2-d for now, DG, triangle elements,
    order 1 for now

    Want a class: fd_fn_space ---> pytential mesh
    such that
        1) only have to compute mesh things once
        2) Ideally data is only a copy

    Plan:
        Mesh:
            1) Prepare information into a Mesh element group
            2) Determine which faces need to be flipped
            3) Create a dictionary [element][new_index] -> [element][old_index]
                3a) If possible, make a data array that [e][new_i]
        Function:
            5) Use the dictionary (or reference data array) to load
               in the value of the function at each node

    Interface:
        firedrake_to_meshmode(fd_fn_space) returns (fn_converter, pyt_mesh)
        fn_converter returns a Discretization in the appropriate space

     NOTE: is pytential mesh what we want?
"""

import pyopencl as cl
cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import numpy as np
from numpy import linalg as la

import firedrake as fd

from meshmode.mesh import MeshElementGroup
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind
from pytential.qbx import QBXLayerPotentialSource


class FiredrakeToMeshmodeConverter:
    def __init__(self, function_space, ambient_dim=None, thresh=1e-5):
        self._thresh = thresh
        """
        :arg function_space: A firedrake.FunctionSpace.
        :arg ambient_dim: By default this is *None*, in which case
            *function_space.mesh().geometric_dimension()* is used.
            If an integer is passed, the mesh must have codimension
            0 or 1.
            NOTE: Why set ambient dimension? To compute layer
                  potentials of a mesh, the co-dimension of the
                  mesh must be exactly 1.
        :arg thresh: Used as threshold for some float equality checks
                     (specifically, those involved in asserting
                     positive ordering of faces)
        NOTE: Only discontinuous lagrange elements are supported
        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """
        # assert that Function Space is using DG elements

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

        self.fd_function_space = function_space

        # The ambient dim to use for the *self._meshmode_mesh* (see below)
        self._ambient_dim = ambient_dim
        if ambient_dim is not None:
            codimension = ambient_dim - \
                function_space.mesh().topological_dimension()
            assert codimension in [0, 1]

        # A *meshmode.mesh.Mesh* object
        self._meshmode_mesh = None

        # An array of size (nelements) with positive entries
        # for each positively oriented element,
        # and a negative entry for each negatively oriented element
        self._orient = None

        # a *pytential.qbx.QBXLayerPotentialSource* object
        # created from a discretization built from *self._meshmode_mesh*
        # using an *InterpolatoryQuadratureSimplexGroupFactory*
        self._qbx = None

        # An array of size (nelements, nnodes) mapping the firedrake index
        # [iel][inode] to the *self._meshmode_mesh* index inode_new
        self._node_reordering = None

        self._flip_matrix = None

    def get_meshmode_mesh(self):
        # construct mesh if not already constructed

        if not self._meshmode_mesh:
            mesh = self.fd_function_space.mesh()
            mesh.init()
            if self._ambient_dim is None:
                self._ambient_dim = mesh.geometric_dimension()
            dim = mesh.topological_dimension()

            # {{{ Construct a SimplexElementGroup
            # Note: not using meshmode.mesh.generate_group
            #       because I need to keep the node ordering
            order = self.fd_function_space.ufl_element().degree()

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
            vertices.resize((self._ambient_dim,)+vertices.shape[1:])

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
            if(nodes.shape[0] < self._ambient_dim):
                nodes = np.resize(nodes, (self._ambient_dim,) + nodes.shape[1:])
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
                if self._ambient_dim == 2:
                    from meshmode.mesh.generation import make_group_from_vertices

                    order_one_grp = make_group_from_vertices(vertices,
                                                             vertex_indices, 1)

                    from meshmode.mesh.processing import (
                            find_volume_mesh_element_group_orientation,
                            flip_simplex_element_group)
                    # This function fails in ambient_dim 3, hence the
                    # separation between ambient_dim 2 and 3
                    self._orient = \
                        find_volume_mesh_element_group_orientation(vertices,
                                                                   order_one_grp)
                else:
                    self._orient = np.zeros(vertex_indices.shape[0])
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
                        self._orient[i] = np.cross(va - v_from, vb - v_from)[-1]
            else:
                raise TypeError("Only geometric dimension 2 is supported")

            from meshmode.mesh import SimplexElementGroup
            group = SimplexElementGroup(order, vertex_indices, nodes, dim=2)

            from meshmode.mesh.processing import flip_simplex_element_group
            group = flip_simplex_element_group(vertices,
                                               group,
                                               self._orient < 0)

            """
            # record node re-ordering
            self._node_reordering = np.zeros(nodes.shape[1:], dtype=np.int32)
            # for each element, record the reordering of the nodes
            for iel in range(nodes.shape[1]):
                new_el_nodes = group.nodes[:, iel, :]
                # transpose so that iterate over nodes and not dimension
                node_to_new_index = {tuple(node): inode for inode, node in
                                     enumerate(new_el_nodes.T)}

                el_nodes = nodes[:, iel, :]
                for inode, node in enumerate(el_nodes.T):
                    self._node_reordering[inode] = node_to_new_index[
                        tuple(node)]

            # }}}
            """

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
            connectivity = self.fd_function_space.finat_element.cell.\
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
            self._meshmode_mesh = Mesh(vertices, groups,
                                       boundary_tags=boundary_tags,
                                       facial_adjacency_groups=facial_adj_grps
                                       )

        return self._meshmode_mesh

    def _determine_flip_matrix(self, grp, grp_flip_flags):
        if self._flip_matrix is None:
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
                    - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

    def get_qbx(self, cl_ctx):
        # build mesh if not built
        if not self._meshmode_mesh:
            self.get_meshmode_mesh()

        if self._qbx is None:
            # FIXME : Assumes order 1
            pre_density_discr = Discretization(
                cl_ctx,
                self._meshmode_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(1))

            # FIXME : Assumes order 1
            # FIXME : Do I have the right thing for the various orders?
            self._qbx = QBXLayerPotentialSource(pre_density_discr,
                                                fine_order=1,
                                                qbx_order=1,
                                                fmm_order=1)
        return self._qbx

    def reorder_nodes(self, nodes, reverse=False):
        """
        :arg nodes: An array representing function values at each of the
                    dofs
        :arg reverse: False if and only if converting firedrake to
                      pytential ordering, else does pytential to firedrake
                      ordering
        """
        if self._meshmode_mesh is None:
            self.get_meshmode_mesh()
        if self._flip_matrix is None:
            self._determine_flip_matrix(
                self._meshmode_mesh.groups[0], self._orient < 0)

        flip_mat = self._flip_matrix
        # reorder data (Code adapted from
        # meshmode.mesh.processing.flip_simplex_element_group)
        if not reverse:
            fd_mesh = self.fd_function_space.mesh()
            vertex_indices = \
                fd_mesh.coordinates.function_space().cell_node_list
        else:
            vertex_indices = self._meshmode_mesh.vertex_indices
            flip_mat = flip_mat.T

        # obtain function data in form [nelements][nunit_nodes]
        data = nodes[vertex_indices]
        # flip nodes that need to be flipped
        data[self._orient < 0] = np.einsum(
            "ij,ej->ei", flip_mat, data[self._orient < 0])

        # convert from [element][unit_nodes] to
        # global node number
        data = data.T.flatten()
        
        return data


    def fd_function_to_array(self, fd_function, dtype=None):
        """
        Returns an array representing the given function on
        *self._meshmode_mesh*
        """
        assert fd_function.function_space() == self.fd_function_space
        return self.reorder_nodes(fd_function.dat.data)

    def array_to_fd_function(self, array, dtype=None):
        """
        :arg array: A type convertible to np.array which represents a function
                    evaluated at the dofs
        :arg dtype: date dtype
        """
        f = fd.Function(self.fd_function_space)
        f.dat.data = self.reorder_nodes(data, reverse=True)

        return f
