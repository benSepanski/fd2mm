"""
    .. _firedrake: http://firedrakeproject.org
    .. _meshmode: http://documen.tician.de/meshmode
    .. _pytential: http://documen.tician.de/pytential

    :mod:`fd2mm` is built to convert `firedrake`_ objects into `meshmode`_, as well
    as providing an easy interface for setting up `pytential`_ operations from
    firedrake (Look at :func:`fd_bind <fd2mm.op.fd_bind>`,
    for more details see :mod:`fd2mm.op`).
"""
from fd2mm.mesh import MeshAnalog
from fd2mm.functionspace import FunctionSpaceAnalog
from fd2mm.function import FunctionAnalog
from fd2mm.op import fd_bind


__all__ = ["MeshAnalog", "FunctionSpaceAnalog", "FunctionAnalog", "fd_bind"]
