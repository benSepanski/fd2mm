import fd2mm.functionspaceimpl as impl
from fd2mm.finat_element import FinatElementAnalog


def FunctionSpaceAnalog(mesh_analog, function_space):
    mesh_analog.init()
    finat_elt_a = FinatElementAnalog(function_space.finat_element)
    function_space_a = impl.FunctionSpaceAnalog(function_space,
                                                mesh_analog,
                                                finat_elt_a)

    return impl.WithGeometryAnalog(function_space, function_space_a, mesh_analog)
