from firedrake import Function
from fd2mm.analogs import Analog
from fd2mm.analogs.functionspaceimpl import WithGeometryAnalog


class CoordinatelessFunctionAnalog(Analog):
    def __init__(self, function, function_space_analog):
        function = function.topological
        function_space_analog = function_space_analog.topological_a

        super(CoordinatelessFunctionAnalog, self).__init__(function)

        self._function_space_a = function_space_analog

    def function_space_a(self):
        return self._function_space_a

    @property
    def topological_a(self):
        return self


class FunctionAnalog(Analog):
    def __init__(self, function, function_space_analog):
        """
            :arg function: A firedrake :class:`Function` or a
                           :class:`CoordinatelessFunctionAnalog`
        """
        # {{{ Check types
        if not isinstance(function_space_analog, WithGeometryAnalog):
            raise TypeError(":arg:`function_space_analog` must be of "
                            "type analogs.WithGeometryAnalog")

        if not isinstance(function, (Function, CoordinatelessFunctionAnalog)):
            raise TypeError(":arg:`function` must be one of "
                            "(:class:`firedrake.funciton.Function`,"
                            " CoordinatelessFunctionAnalog)")
        # }}}

        self._function_space_a = function_space_analog
        if isinstance(function, Function):
            self._data_a = CoordinatelessFunctionAnalog(function,
                                                        function_space_analog)
        else:
            self._data_a = function

    def function_space_a(self):
        return self._function_space_a

    @property
    def topological_a(self):
        return self._data_a

    def as_field(self):
        return self.function_space_a().convert_function(self)

    def set_from_field(self, field):
        order = self.analog().function_space().finat_element.degree
        mesh_order = self.analog().function_space().mesh().\
            coordinates.function_space().finat_element.degree

        if mesh_order > order:
            raise NotImplementedError("Can't convert from higher order"
                                      " mesh in meshmode to lower function"
                                      " mesh in firedrake")

        self.analog().dat.data[:] = self.function_space_a().reorder_nodes(
            field, firedrake_to_meshmode=False)[:]
