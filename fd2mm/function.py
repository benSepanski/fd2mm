import numpy as np

from firedrake import Function

from fd2mm.analog import Analog
from fd2mm.functionspaceimpl import WithGeometryAnalog

from firedrake.petsc import PETSc
converting = PETSc.Log.Stage("Conversion")


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

        super(FunctionAnalog, self).__init__(function)

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
        converting.push()
        field = self.function_space_a().convert_function(self)
        converting.pop()
        return field

    def set_from_field(self, field):
        # Handle 1-D case
        if len(self.analog().dat.data.shape) == 1 and len(field.shape) > 1:
            field = field.reshape(field.shape[1])

        # resample from nodes
        group = self.function_space_a().discretization().groups[0]
        resampled = np.copy(field)
        resampled_view = group.view(resampled)
        resampling_mat = self.function_space_a().resampling_mat(False)
        np.matmul(resampled_view, resampling_mat.T, out=resampled_view)

        # reorder data
        self.analog().dat.data[:] = self.function_space_a().reorder_nodes(
            resampled, firedrake_to_meshmode=False)[:]
