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
