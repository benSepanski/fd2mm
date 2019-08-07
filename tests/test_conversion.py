import pyopencl as cl
import numpy as np
import pytest

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

from firedrake import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, \
    FunctionSpace, VectorFunctionSpace, SpatialCoordinate, sin, exp, pi, \
    Function, as_vector
import fd2mm

TOL = 1e-15


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def mesh(request):
    dim = request.param
    if dim == 1:
        mesh = UnitIntervalMesh(100)
    if dim == 2:
        mesh = UnitSquareMesh(10, 10)
    if dim == 3:
        mesh = UnitCubeMesh(5, 5, 5)
    return mesh


@pytest.fixture
def mesh_analog(mesh):
    return fd2mm.MeshAnalog(mesh)


@pytest.fixture(params=['CG', 'DG'])
def family(request):
    return request.param


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])  # fspace degree
def function_space_analog(request, mesh_analog, family):
    mesh = mesh_analog.analog()
    degree = request.param
    fspace = FunctionSpace(mesh, family, degree)
    return fd2mm.FunctionSpaceAnalog(mesh_analog, fspace)


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])  # fspace degree
def vector_function_space_analog(request, mesh_analog, family):
    print("fd setup")
    mesh = mesh_analog.analog()
    degree = request.param
    vfspace = VectorFunctionSpace(mesh, family, degree)
    print("setup complete")
    return fd2mm.FunctionSpaceAnalog(mesh_analog, vfspace)


def check_idempotent(function_analog):
    r"""
        Make sure that fd->mm->fd and mm->fd->mm
        are idempotent operations, up to *TOL* (in a :math:`\ell^\infty` sense).

        Raises *ValueError*
    """
    original_function = function_analog.analog().copy(deepcopy=True)
    original_field = function_analog.as_field()
    function_analog.set_from_field(original_field)

    new_function = function_analog.analog()
    new_field = function_analog.as_field()

    if np.max(np.abs(original_function.dat.data - new_function.dat.data)) >= TOL:
        raise ValueError("Firedrake->Meshmode->Firedrake is not idempotent")

    if np.max(np.abs(original_field - new_field)) >= TOL:
        raise ValueError("Meshmode->Firedrake->Meshmode is not idempotent")


def test_scalar_idempotent(function_space_analog):
    fspace = function_space_analog.analog()
    xx = SpatialCoordinate(fspace.mesh())

    expressions = {}
    expressions['linear'] = sum([(i+1) * xi for i, xi in enumerate(xx)])
    expressions['quadratic'] = sum([(i+1) * xi**2 for i, xi in enumerate(xx)])
    expressions['sin'] = sum([(i+1) * sin(2.0 * pi * xi) for i, xi in enumerate(xx)])
    expressions['exp'] = sum([(i+1) * exp(xi) for i, xi in enumerate(xx)])

    for key, expr in expressions.items():
        fntn = Function(fspace, name=key).interpolate(expr)
        function_analog = fd2mm.FunctionAnalog(fntn, function_space_analog)
        check_idempotent(function_analog)


def test_vector_idempotent(vector_function_space_analog):
    vfspace = vector_function_space_analog.analog()
    xx = SpatialCoordinate(vfspace.mesh())

    expressions = {}
    expressions['linear'] = [(i+1) * xi for i, xi in enumerate(xx)]
    expressions['quadratic'] = [(i+1) * xi**2 for i, xi in enumerate(xx)]
    expressions['sin'] = [(i+1) * sin(2.0 * pi * xi) for i, xi in enumerate(xx)]
    expressions['exp'] = [(i+1) * exp(xi) for i, xi in enumerate(xx)]

    for key, expr in expressions.items():
        fntn = Function(vfspace, name=key).interpolate(as_vector(expr))
        function_analog = fd2mm.FunctionAnalog(fntn, vector_function_space_analog)
        check_idempotent(function_analog)


def test_coordinate_matching(vector_function_space_analog):
    vfspace = vector_function_space_analog.analog()
    xx = SpatialCoordinate(vfspace.mesh())
    identity_fntn = Function(vfspace).interpolate(xx)

    identity_fntn_analog = fd2mm.FunctionAnalog(identity_fntn,
                                                vector_function_space_analog)
    identity_field = identity_fntn_analog.as_field()

    meshmode_mesh = vector_function_space_analog.meshmode_mesh()

    # Make sure each firedrake node has at least one corresponding meshmode
    # and that the identity field is in fact the identity
    group = meshmode_mesh.groups[0]
    for fd_node in identity_fntn.dat.data:
        none_close = True

        for imm_node in range(group.nnodes):
            if len(identity_field.shape) == 1:
                mm_node = identity_field[imm_node]
            else:
                mm_node = identity_field[:, imm_node]
            # See if is close by
            if np.max(np.abs(fd_node - mm_node)) < TOL:
                none_close = False
                break

        assert not none_close

    # Now convert back and make sure that still have identity
    identity_copy = identity_fntn.copy(deepcopy=True)
    identity_fntn_analog.set_from_field(identity_field)
    assert np.max(np.abs(identity_copy.dat.data - identity_fntn.dat.data)) < TOL
