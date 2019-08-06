from warnings import warn  # noqa
import numpy as np
import numpy.linalg as la


def get_affine_mapping(reference_vects, vects):
    r"""
    Returns (mat, shift),
    a matrix *mat* and vector *shift* which maps the
    *i*th vector in :arg:`v` to the *i*th vector in :arg:`w` by

    ..math::

        A vi + b -> wi, \qquad A = mat, b = shift

    :arg reference_vects: An np.array of *n* vectors of dimension *ref_dim*
    :arg vects: An np.array of *n* vectors of dimension *dim*, with
                *ref_dim* <= *dim*.

        NOTE : Should be shape (ref_dim, nvectors), (dim, nvectors) respectively.

    *mat* will have shape (dim, ref_dim), *shift* will have shape (dim)
    """
    # Make sure both have same number of vectors
    ref_dim, num_vects = reference_vects.shape
    assert num_vects == vects.shape[1]

    # Make sure d1 <= d2 (see docstring)
    dim = vects.shape[0]
    assert ref_dim <= dim

    # If there is only one vector, set M = I, b = vect - reference
    if num_vects == 1:
        mat = np.eye(dim, ref_dim)
        shift = vects[:, 0] - np.matmul(mat, reference_vects[:, 0])
    else:
        ref_span_vects = reference_vects[:, 1:] - reference_vects[:, 0, np.newaxis]
        span_vects = vects[:, 1:] - vects[:, 0, np.newaxis]
        mat = la.solve(ref_span_vects, span_vects)
        shift = -np.matmul(mat, reference_vects[:, 0]) + vects[:, 0]

    return mat, shift


def compute_orientation(mesh, vertices, vertex_indices,
                        normals=None, no_normals_warn=True):
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
    tdim = mesh.topological_dimension()
    gdim = mesh.geometric_dimension()

    orient = None
    if gdim == tdim:
        # TODO: This is probably inefficient design... but some of the
        #       computation needs a :mod:`meshmode` group
        #       right now.

        # We use :mod:`meshmode` to check our orientations
        from meshmode.mesh.generation import make_group_from_vertices
        from meshmode.mesh.processing import \
            find_volume_mesh_element_group_orientation

        group = make_group_from_vertices(vertices,
                                         vertex_indices, 1)
        orient = \
            find_volume_mesh_element_group_orientation(vertices,
                                                       group)

    if tdim == 1 and gdim == 2:
        # In this case we have a 1-surface embedded in 2-space
        orient = np.ones(vertex_indices.shape[0])
        if normals:
            for i, (normal, vertices) in enumerate(zip(
                    np.array(normals), vertices)):
                if np.cross(normal, vertices) < 0:
                    orient[i] = -1.0
        elif no_normals_warn:
            warn("Assuming all elements are positively-oriented.")

    elif tdim == 2 and gdim == 3:
        # In this case we have a 2-surface embedded in 3-space
        orient = mesh.cell_orientations()
        r"""
            Convert (0 \implies negative, 1 \implies positive) to
            (-1 \implies negative, 1 \implies positive)
        """
        orient *= 2
        orient -= np.ones(orient.shape)

    return orient


def reorder_nodes(orient, nodes, flip_matrix, unflip=False):
    """
        :arg orient: An array of shape (nelements) of orientations,
                     >0 for positive, <0 for negative
        :arg nodes: a (nelements, nunit_nodes) or (dim, nelements, nunit_nodes)
                    shaped array of nodes
        :arg flip_matrix: The matrix used to flip each negatively-oriented
                          element
        :arg unflip: If *True*, use transpose of :arg:`flip_matrix` to
                     flip negatively-oriented elements

        flips :arg:`nodes`
    """
    # reorder nodes (Code adapted from
    # meshmode.mesh.processing.flip_simplex_element_group)

    # ( round to int bc applying on integers)
    flip_mat = np.rint(flip_matrix)
    if unflip:
        flip_mat = flip_mat.T

    # flipping twice should be identity
    assert la.norm(
        np.dot(flip_mat, flip_mat)
        - np.eye(len(flip_mat))) < 1e-13

    # }}}

    # {{{ flip nodes that need to be flipped, note that this point we act
    #     like we are in a DG space

    # if a vector function space, nodes array is shaped differently
    if len(nodes.shape) > 2:
        nodes[orient < 0] = np.einsum(
            "ij,ejk->eik",
            flip_mat, nodes[orient < 0])
        # Reshape to [nodes][vector dims]
        nodes = nodes.reshape(
            nodes.shape[0] * nodes.shape[1], nodes.shape[2])
        # pytential wants [vector dims][nodes] not [nodes][vector dims]
        nodes = nodes.T.copy()
    else:
        nodes[orient < 0] = np.einsum(
            "ij,ej->ei",
            flip_mat, nodes[orient < 0])
        # convert from [element][unit_nodes] to
        # global node number
        nodes = nodes.flatten()
