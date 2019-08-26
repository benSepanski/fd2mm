from abc import ABC

__doc__ = """
.. autoclass:: Analog
    :members:

"""


class Analog(ABC):
    """
        Analogs are containers which hold the information
        needed to convert from firedrake objects to
        meshmode objects.
    """
    def __init__(self, analog):
        """
        :arg analog: The firedrake object that this object will
                       be an analog of
        """
        # What this object is an analog of, i.e. the
        # analog of analog is original object
        self._analog = analog

    def analog(self):
        """
            Return the firedrake object that this object is an analog of
            (analog of an analog is the original object).
        """
        return self._analog

    def is_analog(self, obj):
        """
            Return *True* iff this object is an analog of *obj*

            :arg obj: The object to check against
        """
        return self._analog == obj

    def __hash__(self):
        return hash((type(self), self.analog()))

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.analog() == other.analog()

    def __ne__(self, other):
        return not self.__eq__(other)
