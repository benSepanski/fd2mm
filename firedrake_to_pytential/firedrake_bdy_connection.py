import firedrake as fd
import numpy as np


def _get_bdy_mesh(mesh, bdy_ids):
    """
    :arg bdy_ids: iterable of boundary ids of some exterior boundary
        (also may pass an *int* if only one boundary id)

    returns (facet_indices, bdy_mesh)
    where facet_indices[i] is the index in mesh.exterior_facets corresponding to
    the ith cell in bdy_mesh
    """
    if isinstance(bdy_ids, int):
        bdy_ids = [bdy_ids]
    bdy_id_set = set(bdy_ids)

    tdim = mesh.topological_dimension() - 1
    gdim = mesh.geometric_dimension()
    comm = mesh.comm

    fspace = mesh.coordinates.function_space()
    ext_fac = mesh.exterior_facets
    # maps faces to local vertex indices
    connectivity = fspace.finat_element.cell.\
        connectivity[(tdim, 0)]
    markers = ext_fac.markers

    facet_indices = [i for i in range(markers.shape[0]) if markers[i] in bdy_id_set]
    facet_cells = ext_fac.facet_cell[facet_indices]
    local_facet_nr = ext_fac.local_facet_number[facet_indices]

    old_vert_index_to_new = {}
    cells = []
    coords = []
    for cell, facet_nr in zip(facet_cells, local_facet_nr):
        # unpack arguments
        cell, = cell
        facet_nr, = facet_nr

        # Get facet vertex indices
        verts = np.array(connectivity[facet_nr])
        facet = fspace.cell_node_list[cell][verts]

        # convert old vertex indices to new vertex indices
        for vert in facet:
            if vert not in old_vert_index_to_new:
                new = len(old_vert_index_to_new)
                old_vert_index_to_new[vert] = new
                # record coordinates of the vertex
                coords.append(mesh.coordinates.dat.data[vert])

        # make cell (facet with new vertex indices)
        cell = [old_vert_index_to_new[vert] for vert in facet]
        cells.append(cell)

    # convert to array
    cells = np.array(cells)
    coords = np.array(coords)

    plex = fd.mesh._from_cell_list(tdim, cells, coords, comm)
    return facet_indices, fd.mesh.Mesh(plex, dim=gdim)


class FiredrakeBoundaryConnection:
    """
        This class allows passing functions back and forth
        from a Firedrake mesh and one of its exterior boundaries
    """
    def __init__(self, function_space, bdy_ids):
        """
        :arg function_space: should be a function space with
            * Lagrange (CG or DG) elements
            * triangle (or generalization thereof) elements
            * topological and geometric dimension in [2, 3]
            * Either a FunctionSpace, VectorFunctionSpace, or TensorFunctionSpace
                (WARNING: TensorFunctionSpace is untested)

        :arg bdy_ids: iterable of boundary ids of some exterior boundary
            (also may pass an *int* if only one boundary id)
        """
        # function space has right dimension
        gdim = function_space.mesh().geometric_dimension()
        assert gdim in [2, 3]
        tdim = function_space.mesh().topological_dimension()
        assert tdim in [2, 3]
        # function space is on triangle mesh
        assert str(function_space.ufl_cell()) in ['triangle',
            'triangle3D', 'tetrahedron']
        # function space has Lagrange elements
        family = function_space.ufl_element().family()
        assert family in ['Discontinuous Lagrange', 'Lagrange']

        self.from_fspace = function_space
        self.to_fspace = None

        # Get boundary mesh
        mesh = function_space.mesh()
        facet_indices, bdy_mesh = _get_bdy_mesh(mesh, bdy_ids)
        bdy_mesh.init()

        # Create 'to' function space
        degree = function_space.ufl_element().degree()
        if not function_space.shape:
            self.to_fspace = fd.FunctionSpace(bdy_mesh, family, degree=degree)
        elif len(function_space.shape) == 1:
            self.to_fspace = fd.VectorFunctionSpace(bdy_mesh,
                family, degree=degree, dim=function_space.shape[0])
        elif len(function_space.shape) > 1:
            self.to_fspace = fd.TensorFunctionSpace(bdy_mesh,
                family, degree=degree, shape=function_space.shape)

        # Create correspondence between nodes
        # bdy_node_to_global[i] is the index of the node in
        # from_fspace corresponding to the ith node in the
        # to_fspace
        self.bdy_node_to_global = {}

        ext_fac = mesh.exterior_facets
        # maps local facet number to local node numbers
        # of nodes on that facet
        dofs = function_space.finat_element.entity_support_dofs()[tdim-1]

        for i, facet_index in enumerate(facet_indices):
            cell_nr, = ext_fac.facet_cell[facet_index]
            facet_nr, = ext_fac.local_facet_number[facet_index]
            # cell-local indexes of nodes on facet
            facet_nodes_local = np.array(dofs[facet_nr], dtype=np.int32)
            # global indexes of nodes on facet
            facet_nodes = function_space.cell_node_list[cell_nr][facet_nodes_local]

            # FIXME : This assumes everything is ordered the same, is
            #         this right?
            facet_nodes_in_bdy_mesh = self.to_fspace.cell_node_list[i]
            for nr, node in enumerate(facet_nodes_in_bdy_mesh):
                if node not in self.bdy_node_to_global:
                    self.bdy_node_to_global[node] = facet_nodes[nr]

        self.bdy_node_to_global = [
            self.bdy_node_to_global[i] for i in range(len(self.bdy_node_to_global))]
        self.bdy_node_to_global = np.array(self.bdy_node_to_global, dtype=np.int32)

    def __call__(self, function):
        """
        :arg function: Depending on which function space
            function lives in, converts function to the other.
            Note that if converting from boundary to the whole mesh, it assumes
            the function is 0 away from the boundary
        """
        if function.function_space() == self.from_fspace:
            new_function = fd.Function(self.to_fspace)
            for to_ind, from_ind in enumerate(self.bdy_node_to_global):
                new_function.dat.data[to_ind] = function.dat.data[from_ind]
        elif function.function_space() == self.to_fspace:
            fntn_shape = (self.from_fspace.dof_count,) + self.from_fspace.shape
            new_function = fd.Function(self.from_fspace, val=np.zeros(fntn_shape))
            new_function.dat.data[self.bdy_node_to_global] = function.dat.data
        else:
            raise ValueError("Function must be from either *self.from_fspace*"
            " or *self.to_fspace")

        return new_function
