from abc import ABC


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


from fd2mm.analogs.mesh import MeshAnalog, MeshAnalogNearBdy, \
    MeshAnalogOnBdy
from fd2mm.analogs.functionspace import FunctionSpaceAnalog
from fd2mm.analogs.function import FunctionAnalog
__all__ = ["MeshAnalog", "MeshAnalogNearBdy", "MeshAnalogOnBdy",
           "FunctionSpaceAnalog"]