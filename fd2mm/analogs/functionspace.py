from fd2mm.analogs.finat_element import FinatElementAnalog
import fd2mm.analogs.functionspaceimpl as impl


def FunctionSpaceAnalog(mesh_analog, function_space):
    finat_elt_a = FinatElementAnalog(function_space.finat_element)
    function_space_a = impl.FunctionSpaceAnalog(function_space,
                                                mesh_analog,
                                                finat_elt_a)

    return impl.WithGeometryAnalog(function_space, function_space_a, mesh_analog)
