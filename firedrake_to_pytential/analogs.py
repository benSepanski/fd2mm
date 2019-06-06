from warnings import warn
from abc import ABC, abstractmethod
import six

import numpy as np
from numpy import linalg as la

from FIAT.reference_element import Simplex
from firedrake.functionspaceimpl import WithGeometry

from modepy import tools


def _get_affine_mapping(v, w):
    """
    Returns (A, b),
    a matrix A and vector b which maps the *i*th vector in :arg:`v`
    to the *i*th vector in :arg:`w` by
    Avi + b -> wi

    :arg v: An np.array of *n* vectors of dimension *d*
    :arg w: An np.array of *n* vectors of dimension *d*

        NOTE : Both should be (d, nvectors)
    """
    assert v.shape[0] == w.shape[0]

    if v.shape[0] == 1:
        A = np.eye(v.shape[0])
        b = w[0] - v[0]
    else:
        v_span_vects = v[:, 1:] - v[:, 0, np.newaxis]
        w_span_vects = w[:, 1:] - w[:, 0, np.newaxis]
        A = np.linalg.solve(v_span_vects, w_span_vects)
        b = np.matmul(A, -v[:, 0]) + w[:, 0]
    return A, b


class Analog(ABC):
    """
        Analogs are containers which hold the information
        needed to convert between :mod:`firedrake` and
        :mod:`meshmode`.
    """
    def __init__(self, analog):
        # What this object is an analog of, i.e. the
        # analog of analog is original object
        self._analog = analog

    def analog(self):
        """
            Return what this is an analog of, i.e. the
            analog of this analog
        """
        return self._analog

    def is_analog(self, obj):
        return self._analog == obj

    def __hash__(self):
        return hash((type(self), self.analog()))

    def __eq__(self, other):
        return (isinstance(self, type(other)) and self.analog() == other.analog())

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __getattr__(self, attr):
        try:
            return self.analog().__getattribute__(attr)
        except AttributeError:
            return self.analog().__getattr__(attr)

        raise AttributeError


class SimplexCellAnalog(Analog):
    """
        cell analog, for simplices only
    """
    def __init__(self, cell):
        """
            :arg cell: a :mod:`FIAT` :class:`reference_element`
        """
        # Ensure this cell is actually a simplex
        assert isinstance(cell, Simplex)

        super(SimplexCellAnalog, self).__init__(cell)

        reference_vertices = np.array(cell.vertices)

        dim = reference_vertices.shape[0] - 1
        # Stored as (nunit_vertices, dim)
        self._unit_vertices = tools.unit_vertices(dim)

        # Maps firedrake reference nodes to :mod:`meshmode`
        # unit nodes
        self._A, self._b = _get_affine_mapping(reference_vertices.T,
                                               self._unit_vertices.T)

    def make_points(self, dim, entity_id, order):
        """
            as called by a cell in firedrake, but converted
            to :mod:`modepy` unit nodes
        """
        points = self.analog().make_points(dim, entity_id, order)
        if not points:
            return points
        points = np.array(points)
        # Points is (nvertices, dim) so have to transpose
        return (np.matmul(self._A, points.T) + self._b).T


class FinatElementAnalog(Analog):
    def __init__(self, finat_element, cell_analog=None):
        """
            :arg finat_element: A :mod:`finat` fiat element
            :arg cell_analog: Either a :class:`SimplexCellAnalog` associated to the
                              :attr:`cell` of :arg:`finat_element`, or *None*, in
                              which case a :class:`SimplexCellAnalog` is constructed.
        """

        self._unit_nodes = None
        self._barycentric_unit_nodes = None
        self._flip_matrix = None

        if cell_analog is None:
            # Construct cell analog if needed
            cell_analog = SimplexCellAnalog(self, finat_element.cell)
        else:
            # Else make sure the cell analog is an analog the right cell!
            assert cell_analog.analog() == finat_element.cell

        self._cell_analog = cell_analog
        super(FinatElementAnalog, self).__init__(finat_element)

    def unit_nodes(self):
        """
            gets unit nodes (following :mod:`modepy` rules for the reference simplex)
            as (dim, nunit_nodes)
        """
        if self._unit_nodes is None:
            node_nr_to_coords = {}

            # {{{ Get unit nodes (in :mod:`meshmode` coordinates)
            for dim, element_nrs in six.iteritems(
                    self.entity_support_dofs()):
                for element_nr, node_list in six.iteritems(element_nrs):
                    pts_on_element = self._cell_analog.make_points(
                        dim, element_nr, self.degree)
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

    def barycentric_unit_nodes(self):
        """
            Gets unit nodes in barycentric coordinates
            as (dim, nunit_nodes)
        """
        if self._barycentric_unit_nodes is None:
            # unit vertices is (nunit_vertices, dim), change to
            # (dim, nunit_vertices)
            unit_vertices = self._cell_analog._unit_vertices.T.copy()
            r"""
                If the vertices of the unit simplex are

                ..math::

                v_0,\dots, v_n

                we want to write some vector x as

                ..math::

                \sum b_i v_i, \qquad \sum b_i = 1

                In particular, we have

                ..math::

                b_0 = 1 - \sum b_i
                \implies
                x - b_0 = \sum b_i(v_i - v_0)

            """
            # A <- [v_1 - v_0, v_2 - v_0, \dots, v_d - v_0]
            A = unit_vertices[:, 1:] - unit_vertices[:, 0, np.newaxis]
            # A <- A^{-1}
            A = np.linalg.inv(A)

            # [node_1 - v_0, node_2 - v_0, \dots, node_n - v_0]
            shifted_unit_nodes = self.unit_nodes() - unit_vertices[:, 0, np.newaxis]

            # b_1,\dots, b_n are computed for each unit node
            # shape (dim, unit_nodes)
            bary_nodes = np.matmul(A, shifted_unit_nodes)

            dim, nunit_nodes = self.unit_nodes().shape
            self._barycentric_unit_nodes = np.ones((dim + 1, nunit_nodes))

            # compute b_0 for each unit node
            self._barycentric_unit_nodes[0] -= np.einsum("ij->j", bary_nodes)
            self._barycentric_unit_nodes[1:] = bary_nodes

        return self._barycentric_unit_nodes

    def flip_matrix(self):
        if self._flip_matrix is None:
            # This is very similar to :mod:`meshmode` in processing.py
            # the function :function:`from_simplex_element_group`, but
            # we needed to use firedrake nodes

            from modepy.tools import barycentric_to_unit, unit_to_barycentric

            # Generate a resampling matrix that corresponds to the
            # first two barycentric coordinates being swapped.

            bary_unit_nodes = unit_to_barycentric(self.unit_nodes())

            flipped_bary_unit_nodes = bary_unit_nodes.copy()
            flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
            flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
            flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

            from modepy import resampling_matrix, simplex_best_available_basis
            dim = self.cell.get_dimension()

            flip_matrix = resampling_matrix(
                simplex_best_available_basis(dim, self.degree),
                flipped_unit_nodes, self.unit_nodes())

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

        return self._flip_matrix


class MeshAnalog(Analog):
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

        ..atribute:: _group

            A :mod:`meshmode` :class:`MeshElementGroup` used to construct
            orientations and facial adjacency
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

        self.normals = normals
        self.no_normals_warn = no_normals_warn

        # Create attributes to be computed later
        self._vertices = None
        self._vertex_indices = None
        self._orient = None
        self._facial_adjacency_groups = None
        self._boundary_tags = None
        self._group = None

    def vertices(self):
        """
        An array holding the coordinates of the vertices
        """

        if self._vertices is None:
            # Nb: coordinates is an attirbute of the firedrake mesh
            self._vertices = np.real(self.coordinates.dat.data)

            #:mod:`meshmode` wants [ambient_dim][nvertices], but for now we
            #write it as [geometric dim][nvertices]
            self._vertices = self._vertices.T.copy()

        return self._vertices

    def vertex_indices(self):
        """
            (analog to mesh.coordinates.function-space().cell_node_list)

            <from :mod:`meshmode` docs:>
            An array of (nelements, ref_element.nvertices)
            of (mesh-wide) vertex indices
        """
        if self._vertex_indices is None:
            # Nb: coordinates is an attribute of the firedrake mesh
            self._vertex_indices = np.copy(
                self.coordinates.function_space().cell_node_list)

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
                # Nb: cell_orientations is an attribute of the firedrake mesh
                self._orient = self.cell_orientations().dat.data.astype(np.float64)
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
            # Nb : coordinates is an attribute of the firedrake mesh
            # maps faces to local vertex indices
            connectivity = self.coordinates.function_space().finat_element.cell.\
                connectivity[(self.topological_dimension() - 1, 0)]

            # Nb: exterior_facets is an attribute of the firedrake mesh
            efacets = self.exterior_facets

            if efacets.facet_cell.size == 0 and self._tdim >= self._gdim:
                warn("No exterior facets listed in"
                     " <mesh>.exterior_facets.facet_cell. In particular, NO BOUNDARY"
                     " information is tagged.")

            for i, (icell, ifac) in enumerate(zip(
                    efacets.facet_cell, efacets.local_facet_dat.data)):
                ifac = ifac
                # unpack arguments
                icell, = icell
                # record face vertex indices to tag map
                facet_indices = connectivity[ifac]
                fvi = frozenset(self.vertex_indices()[icell]
                                [list(facet_indices)])
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

            efacets = self.exterior_facets  # Attribute of the firedrake mesh
            if efacets.unique_markers is not None:
                for tag in efacets.unique_markers:
                    self._boundary_tags.append(tag)
            self._boundary_tags = tuple(self._boundary_tags)

        return self._boundary_tags


class FunctionSpaceAnalog(Analog):
    """
        NOTE : This is a special case of an Analog, because
               firedrake has more information than we need
               in a FunctionSpace. In particular,
               :function:`__getattr__` and :function:`is_analog`
               are overwritten, as they are not what you normally
               would expect.

        NOTE : The BEST space analog to look at, to get an idea of how
               things work, is DGFunctionSpaceAnalog. For CG spaces,
               there are nodes being created/destroyed, since everything
               in :mod:`pytential` uses discontinuous things
    """

    def __init__(self, function_space=None,
                 cell_analog=None, finat_element_analog=None, mesh_analog=None):
        """
            :arg function_space: Either a :mod:`firedrake` function space or *None*.
                                 One should note that this function space is NOT
                                 stored in the object. This is so that different
                                 function spaces (e.g. a function space and a vector
                                 function space of the same degree on the same mesh)
                                 can share an Analog (see the class documentation)

            :arg:`mesh_analog`, :arg:`finat_element_analog`, and :arg:`cell_analog`
            are required if :arg:`function_space` is *None*.
            If the function space is known a priori, these are only passed in to
            avoid duplication of effort (e.g. if you are making multiple
            :class:`FunctionSpaceAnalog` objects on the same mesh, there's no
            reason for them both to construct :class:`MeshAnalogs`of that mesh).
        """

        # Construct analogs if necessary
        if function_space is not None:
            if cell_analog is None:
                cell_analog = SimplexCellAnalog(function_space.finat_element.cell)

            if finat_element_analog is None:
                finat_element_analog = FinatElementAnalog(
                    function_space.finat_element, cell_analog=cell_analog)

            if mesh_analog is None:
                mesh_analog = MeshAnalog(function_space.mesh())

        # Make sure the analogs are of the appropriate types
        assert isinstance(cell_analog, SimplexCellAnalog)
        assert isinstance(finat_element_analog, FinatElementAnalog)
        assert isinstance(mesh_analog, MeshAnalog)

        # Make sure the analogs are compatible

        if not cell_analog.is_analog(finat_element_analog.cell):
            raise ValueError("Finat element analog and cell analog must refer"
                             " to the same cell")
        if function_space is not None:
            assert cell_analog.is_analog(function_space.finat_element.cell)
            assert finat_element_analog.is_analog(function_space.finat_element)
            assert mesh_analog.is_analog(function_space.mesh())

        # Call to super
        super(FunctionSpaceAnalog, self).__init__(
            (cell_analog.analog(), finat_element_analog.analog(),
             mesh_analog.analog()))

        self._nodes = None
        self._meshmode_mesh = None
        self._fd_to_mesh_reordering = None
        self._mesh_to_fd_reordering = None

        self._mesh_analog = mesh_analog
        self._cell_analog = cell_analog
        self._finat_element_analog = finat_element_analog

    def __getattr__(self, attr):
        for obj in self.analog():
            try:
                return obj.__getattribute__(attr)
            except AttributeError:
                try:
                    return obj.__getattr__(attr)
                except AttributeError:
                    pass
        raise AttributeError

    def unit_nodes(self):
        """
            get unit nodes as an *np.ndarray* of shape (dim, nunit nodes)
        """
        return self._finat_element_analog.unit_nodes()

    def degree(self):
        """
            return degree of FiNAT element
        """
        return self._finat_element_analog.degree

    def is_analog(self, obj):
        if not isinstance(obj, WithGeometry):
            return False

        mesh = obj.mesh()
        finat_element = obj.finat_element
        cell = finat_element.cell

        return (cell, finat_element, mesh) == self.analog()

    @abstractmethod
    def _reordering_array(self, firedrake_to_meshmode):
        """
        Returns a *np.array* that can reorder the data by composition,
        see :function:`reorder_nodes` below
        """

    @abstractmethod
    def meshmode_mesh(self):
        """
            return a :mod:`meshmode` :class:`Mesh` which
            corresponds to the :mod:`firedrake` mesh of the
            function space this object is an analog of
        """

    def reorder_nodes(self, nodes, firedrake_to_meshmode=True):
        """
        :arg nodes: An array representing function values at each of the
                    dofs
        :arg firedrake_to_meshmode: *True* iff firedrake->meshmode, *False*
            if reordering meshmode->firedrake
        """
        reordered_nodes = nodes[self._reordering_array(firedrake_to_meshmode)]
        # handle vector spaces
        if len(nodes.shape) > 1:
            reordered_nodes = reordered_nodes.T.copy()

        return reordered_nodes


class DGFunctionSpaceAnalog(FunctionSpaceAnalog):
    """
        A function space analog for DG function spaces
    """

    def __init__(self, function_space=None,
                 cell_analog=None, finat_element_analog=None, mesh_analog=None):

        super(DGFunctionSpaceAnalog, self).__init__(
            function_space=function_space,
            cell_analog=cell_analog,
            finat_element_analog=finat_element_analog,
            mesh_analog=mesh_analog)

        from finat.fiat_elements import DiscontinuousLagrange
        if not isinstance(finat_element_analog.analog(),
                          DiscontinuousLagrange):
            raise ValueError("Must use Discontinuous Lagrange elements")

    def _reordering_array(self, firedrake_to_meshmode):
        # See if need to compute array
        order = None
        if firedrake_to_meshmode and self._fd_to_mesh_reordering is None or \
                (not firedrake_to_meshmode and self._mesh_to_fd_reordering is None):

            self.meshmode_mesh()  # To make sure things aren't left uncomputed
            num_nodes = self._nodes.shape[1] * self._nodes.shape[2]
            order = np.arange(num_nodes)

        # Compute permutation if not already done
        if order is not None:
            # {{{ reorder nodes (Code adapted from
            # meshmode.mesh.processing.flip_simplex_element_group)

            # obtain function data in form [nelements][nunit_nodes]
            # and get flip mat
            # ( round to int bc applying on integers)
            flip_mat = np.rint(self._finat_element_analog.flip_matrix())
            if not firedrake_to_meshmode:
                flip_mat = flip_mat.T

            # flipping twice should be identity
            assert np.linalg.norm(
                np.dot(flip_mat, flip_mat)
                - np.eye(len(flip_mat))) < 1e-13

            # else reshape new_order so that can be re-ordered
            nunit_nodes = self.unit_nodes().shape[1]
            new_order = order.reshape(
                (order.shape[0]//nunit_nodes, nunit_nodes) + order.shape[1:])

            # flip nodes that need to be flipped

            orient = self._mesh_analog.orientations()
            # if a vector function space, new_order array is shaped differently
            if len(order.shape) > 1:
                new_order[orient < 0] = np.einsum(
                    "ij,ejk->eik",
                    flip_mat, new_order[orient < 0])
                new_order = new_order.reshape(
                    new_order.shape[0] * new_order.shape[1], new_order.shape[2])
                # pytential wants [vector dims][nodes] not [nodes][vector dims]
                new_order = new_order.T.copy()
            else:
                new_order[orient < 0] = np.einsum(
                    "ij,ej->ei",
                    flip_mat, new_order[orient < 0])
                # convert from [element][unit_nodes] to
                # global node number
                new_order = new_order.flatten()

            # }}}

            if firedrake_to_meshmode:
                self._fd_to_mesh_reordering = new_order
            else:
                self._mesh_to_fd_reordering = new_order

        # Return the appropriate array
        if firedrake_to_meshmode:
            return self._fd_to_mesh_reordering
        else:
            return self._mesh_to_fd_reordering

    def meshmode_mesh(self):
        if self._meshmode_mesh is None:

            unit_nodes = self.unit_nodes()

            topological_dim = self._mesh_analog.topological_dimension()
            vertex_indices = self._mesh_analog.vertex_indices()
            vertices = self._mesh_analog.vertices()
            orient = self._mesh_analog.orientations()

            # {{{ Compute nodes

            ambient_dim = self._mesh_analog.geometric_dimension()
            nelements = vertex_indices.shape[0]
            nunit_nodes = unit_nodes.shape[1]
            bary_unit_nodes = self._finat_element_analog.barycentric_unit_nodes()

            nodes = np.zeros((ambient_dim, nelements, nunit_nodes))
            # NOTE : This relies on the fact that for DG elements, nodes
            #        in firedrake ordered nicely, i.e.
            # [0, 1, ...,nunodes-1], [nunodes, nunodes+1, ... 2nunodes-1],
            # ...
            for i, indices in enumerate(vertex_indices):
                elt_coords = np.zeros((ambient_dim, len(indices)))
                for j in range(elt_coords.shape[1]):
                    elt_coords[:, j] = vertices[:, indices[j]]

                nodes[:, i, :] = np.matmul(elt_coords, bary_unit_nodes)[:, :]

            # }}}

            from meshmode.mesh import SimplexElementGroup
            group = SimplexElementGroup(self.degree(), vertex_indices, nodes,
                                        dim=topological_dim, unit_nodes=unit_nodes)

            from meshmode.mesh.processing import flip_simplex_element_group
            group = flip_simplex_element_group(vertices,
                                               group,
                                               orient < 0)

            groups = [group]

            facial_adj_grps = self._mesh_analog.facial_adjacency_groups()
            boundary_tags = self._mesh_analog.boundary_tags()

            from meshmode.mesh import Mesh
            self._meshmode_mesh = Mesh(vertices, groups,
                                       boundary_tags=boundary_tags,
                                       facial_adjacency_groups=facial_adj_grps)
            self._nodes = nodes

        return self._meshmode_mesh


class CGFunctionSpaceAnalog(FunctionSpaceAnalog):
    """
        A function space analog for CG function spaces
    """

    def __init__(self, function_space=None,
                 cell_analog=None, finat_element_analog=None, mesh_analog=None):

        super(CGFunctionSpaceAnalog, self).__init__(
            function_space=function_space,
            cell_analog=cell_analog,
            finat_element_analog=finat_element_analog,
            mesh_analog=mesh_analog)

        from finat.fiat_elements import Lagrange
        if not isinstance(finat_element_analog.analog(), Lagrange):
            raise ValueError("Must use Lagrange elements")

        warn("Careful! :mod:`meshmode` uses all DG elements, so"
             " we convert CG -> DG (fd->pytential) [OKAY] and DG -> CG"
             " (pytential->fd) [DANGEROUS--ONLY DO IF YOU KNOW RESULT"
             " WILL BE CONTINUOUS]")

        # If we weren't givn a function space, we'll compute these later
        self._cell_node_list = None
        self._num_fdnodes = None

        # If we were given a function space, no need to compute them again later!
        if function_space is not None:
            self._cell_node_list = function_space.cell_node_list
            self._num_fdnodes = np.max(self._cell_node_list) + 1

    def _reordering_array(self, firedrake_to_meshmode):
        # See if need to compute array
        order = None
        if firedrake_to_meshmode and self._fd_to_mesh_reordering is None:
            self.meshmode_mesh()  # To make sure things aren't left uncomputed
            order = np.arange(self._num_fdnodes)
        elif not firedrake_to_meshmode and self._mesh_to_fd_reordering is None:
            self.meshmode_mesh()  # To make sure things aren't left uncomputed
            num_meshmode_nodes = self._nodes.shape[1] * self._nodes.shape[2]
            order = np.arange(num_meshmode_nodes)

        # Compute permutation if not already done
        if order is not None:
            flip_mat = np.rint(self._finat_element_analog.flip_matrix())
            if not firedrake_to_meshmode:
                flip_mat = flip_mat.T

            # flipping twice should be identity
            assert np.linalg.norm(
                np.dot(flip_mat, flip_mat)
                - np.eye(len(flip_mat))) < 1e-13

            # re-size if firedrake to meshmode
            if firedrake_to_meshmode:
                new_order = order[self._cell_node_list]
            # else reshape new_order so that can be re-ordered
            else:
                nunit_nodes = self.unit_nodes().shape[1]
                new_order = order.reshape(
                    (order.shape[0]//nunit_nodes, nunit_nodes) + order.shape[1:])

            # flip nodes that need to be flipped

            orient = self._mesh_analog.orientations()
            # if a vector function space, new_order array is shaped differently
            if len(order.shape) > 1:
                new_order[orient < 0] = np.einsum(
                    "ij,ejk->eik",
                    flip_mat, new_order[orient < 0])
                new_order = new_order.reshape(
                    new_order.shape[0] * new_order.shape[1], new_order.shape[2])
                # pytential wants [vector dims][nodes] not [nodes][vector dims]
                new_order = new_order.T.copy()
            else:
                new_order[orient < 0] = np.einsum(
                    "ij,ej->ei",
                    flip_mat, new_order[orient < 0])
                # convert from [element][unit_nodes] to
                # global node number
                new_order = new_order.flatten()

            # }}}

            # Resize new_order if going meshmode->firedrake
            if not firedrake_to_meshmode:
                # FIXME : This is done EXTREMELY lazily
                newnew_order = np.zeros(self._num_fdnodes, dtype=np.int32)
                # NOTE: This relies on how we order nodes for DG, i.e.
                #       the way you'd expect: [0,1,..,n], [n+1,...]
                pyt_ndx = 0
                for nodes in self._cell_node_list:
                    for fd_index in nodes:
                        newnew_order[fd_index] = new_order[pyt_ndx]
                        pyt_ndx += 1

                new_order = newnew_order

            if firedrake_to_meshmode:
                self._fd_to_mesh_reordering = new_order
            else:
                self._mesh_to_fd_reordering = new_order

        # Return the appropriate array
        if firedrake_to_meshmode:
            return self._fd_to_mesh_reordering
        else:
            return self._mesh_to_fd_reordering

    def meshmode_mesh(self):
        if self._meshmode_mesh is None:

            # {{{ Construct firedrake cell node list if not already constructed
            if self._cell_node_list is None:
                entity_dofs = self._finat_element_analog.analog().entity_dofs()
                mesh = self._mesh_analog.analog()
                nodes_per_entity = tuple(mesh.make_dofs_per_plex_entity(entity_dofs))

                # FIXME : Allow for real tensor products
                from firedrake.functionspacedata import get_global_numbering
                global_numbering = get_global_numbering(mesh,
                                                        (nodes_per_entity, False))
                self._cell_node_list = mesh.make_cell_node_list(global_numbering,
                                                                entity_dofs, None)
                self._num_fdnodes = np.max(self._cell_node_list) + 1

            # }}}

            unit_nodes = self.unit_nodes()

            topological_dim = self._mesh_analog.topological_dimension()
            vertex_indices = self._mesh_analog.vertex_indices()
            vertices = self._mesh_analog.vertices()
            orient = self._mesh_analog.orientations()

            # {{{ Compute nodes

            ambient_dim = self._mesh_analog.geometric_dimension()
            nelements = vertex_indices.shape[0]
            nunit_nodes = unit_nodes.shape[1]
            bary_unit_nodes = self._finat_element_analog.barycentric_unit_nodes()

            nodes = np.zeros((ambient_dim, nelements, nunit_nodes))

            for i, indices in enumerate(vertex_indices):
                elt_coords = np.zeros((ambient_dim, len(indices)))
                for j in range(elt_coords.shape[1]):
                    elt_coords[:, j] = vertices[:, indices[j]]

                # NOTE : Here, we are in effect 'creating' nodes, since some
                #        nodes that were shared along boundaries are now treated
                #        as independent
                #
                #        In particular, this node numbering will be different
                #        than firedrake's!
                nodes[:, i, :] = np.matmul(elt_coords, bary_unit_nodes)[:, :]

            # }}}

            # {{{ Construct mesh and store reordered nodes

            from meshmode.mesh import SimplexElementGroup
            group = SimplexElementGroup(self.degree(), vertex_indices, nodes,
                                        dim=topological_dim, unit_nodes=unit_nodes)

            from meshmode.mesh.processing import flip_simplex_element_group
            group = flip_simplex_element_group(vertices,
                                               group,
                                               orient < 0)

            groups = [group]

            boundary_tags = self._mesh_analog.boundary_tags()
            facial_adj_grps = self._mesh_analog.facial_adjacency_groups()

            from meshmode.mesh import Mesh
            self._meshmode_mesh = Mesh(vertices, groups,
                                       boundary_tags=boundary_tags,
                                       facial_adjacency_groups=facial_adj_grps)
            self._nodes = nodes

            # }}}

        return self._meshmode_mesh
