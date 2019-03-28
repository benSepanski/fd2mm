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

    facet_indices = [i for i, marker in enumerate(markers) if marker in bdy_id_set]
    facet_cells = ext_fac.facet_cell[facet_indices]
    local_facet_nr = ext_fac.local_facet_number[facet_indices]

    verts_used = set()
    cells = []
    coords = []
    for cell, facet_nr in zip(facet_cells, local_facet_nr):
        # unpack arguments
        cell, = cell
        facet_nr, = facet_nr

        # Get facet vertex indices
        verts = np.array(connectivity[facet_nr])
        facet = fspace.cell_node_list[cell][verts]

        # record vertices used
        for vert in facet:
            if vert not in verts_used:
                verts_used.add(vert)

        # copy old facet vert indices
        cells.append(facet)

    # Create new vertex indices
    new_vert_index_to_old = list(verts_used)
    new_vert_index_to_old.sort()  # to keep same relative order of verts

    old_vert_index_to_new = {old: new for new, old in
        enumerate(new_vert_index_to_old)}

    # Convert cells to new vertex indices
    cells = [[old_vert_index_to_new[i] for i in cell] for cell in cells]

    # Create coordinate array
    coords = [mesh.coordinates.dat.data[old_vert_index] for old_vert_index in
        new_vert_index_to_old]

    # convert to array
    cells = np.array(cells)
    coords = np.array(coords)

    plex = fd.mesh._from_cell_list(tdim, cells, coords, comm)

    return facet_indices, fd.mesh.Mesh(plex, dim=gdim, reorder=False)


# WARNING: This ONLY WORKS for DEGREE 1, GDIM 2
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

        # Get node coordinates
        bdy_xx = fd.SpatialCoordinate(bdy_mesh)
        xx = fd.SpatialCoordinate(mesh)
        todim = fd.VectorFunctionSpace(bdy_mesh, family, degree=degree, dim=gdim)
        fromdim = fd.VectorFunctionSpace(mesh, family, degree=degree, dim=gdim)
        bdy_coords = fd.Function(todim).interpolate(bdy_xx).dat.data
        coords = fd.Function(fromdim).interpolate(xx).dat.data
        # Create node correspondence map
        for i, facet_index in enumerate(facet_indices):
            cell_nr, = ext_fac.facet_cell[facet_index]
            facet_nr, = ext_fac.local_facet_number[facet_index]
            # cell-local indexes of nodes on facet
            facet_nodes_local = np.array(dofs[facet_nr], dtype=np.int32)
            # global indexes of nodes on facet
            facet_nodes = function_space.cell_node_list[cell_nr][facet_nodes_local]

            facet_nodes_in_bdy_mesh = self.to_fspace.cell_node_list[i]
            for node in facet_nodes_in_bdy_mesh:
                if node not in self.bdy_node_to_global:
                    # Because Mesh was created with reorder=False,
                    # each facet refers to the correct set of nodes. However,
                    # These may be stored in a different order
                    node_coords = bdy_coords[node]
                    corr_node = facet_nodes[0]
                    dist = np.linalg.norm(node_coords - coords[corr_node])
                    # Check all possible nodes in the whole mesh
                    # (i.e. those of the corresponding facet), 
                    # and pick the closest as the corresponding node
                    for possible_node in facet_nodes[1:]:
                        new_dist = np.linalg.norm(
                            node_coords - coords[possible_node])
                        if new_dist < dist:
                            dist = new_dist
                            corr_node = possible_node
                    self.bdy_node_to_global[node] = corr_node

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
            " or *self.to_fspace*")

        return new_function
