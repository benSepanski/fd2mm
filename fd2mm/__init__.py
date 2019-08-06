from abc import ABC

import fd2mm.functionspaceimpl as impl

from fd2mm.mesh import MeshTopologyAnalog, MeshGeometryAnalog
from fd2mm.finat_element import FinatElementAnalog
from fd2mm.function import CoordinatelessFunctionAnalog


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

    def is_analog(self, obj, **kwargs):
        """
            Return *True* iff this object is an analog of :arg:`obj`
        """
        return self._analog == obj

    def __hash__(self):
        return hash((type(self), self.analog()))

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.analog() == other.analog()

    def __ne__(self, other):
        return not self.__eq__(other)


def MeshAnalog(mesh):
    coords_fspace = mesh.coordinates.function_space()

    topology_a = MeshTopologyAnalog(mesh)
    finat_elt_a = FinatElementAnalog(coords_fspace.finat_element)

    coords_fspace_a = FunctionSpaceAnalog(coords_fspace, topology_a, finat_elt_a)
    coordinates_analog = CoordinatelessFunctionAnalog(mesh.coordinates,
                                                      coords_fspace_a)

    return MeshGeometryAnalog(mesh, coordinates_analog)


def FunctionSpaceAnalog(mesh_analog, function_space):
    finat_elt_a = FinatElementAnalog(function_space.finat_element)
    function_space_a = impl.FunctionSpaceAnalog(function_space,
                                                mesh_analog,
                                                finat_elt_a)

    return impl.WithGeometryAnalog(function_space, function_space_a, mesh_analog)


from fd2mm.function import FunctionAnalog
from fd2mm.op import fd_bind

__all__ = ["MeshAnalog", "FunctionSpaceAnalog", "FunctionAnalog",
           "fd_bind"]
