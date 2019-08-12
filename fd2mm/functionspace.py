import fd2mm.functionspaceimpl as impl
from fd2mm.finat_element import FinatElementAnalog


def FunctionSpaceAnalog(cl_ctx, mesh_analog, function_space):
    mesh_analog.init(cl_ctx)
    finat_elt_a = FinatElementAnalog(function_space.finat_element)
    function_space_a = impl.FunctionSpaceAnalog(function_space,
                                                mesh_analog,
                                                finat_elt_a)

    return impl.WithGeometryAnalog(cl_ctx, function_space, function_space_a, mesh_analog)
