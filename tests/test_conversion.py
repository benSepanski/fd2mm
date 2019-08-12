import pyopencl as cl
import numpy as np
import pytest

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

from firedrake import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, \
    FunctionSpace, VectorFunctionSpace, SpatialCoordinate, sin, exp, pi, \
    Function, as_vector
import fd2mm

TOL_EXP = 12
TOL = 10**-TOL_EXP


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
    mesh_a = fd2mm.MeshAnalog(mesh)
    return mesh_a


@pytest.fixture(params=['CG', 'DG'])
def family(request):
    return request.param


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])  # fspace degree
def function_space_analog(request, mesh_analog, family):
    mesh = mesh_analog.analog()
    degree = request.param
    fspace = FunctionSpace(mesh, family, degree)
    return fd2mm.FunctionSpaceAnalog(cl_ctx, mesh_analog, fspace)


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])  # fspace degree
def vector_function_space_analog(request, mesh_analog, family):
    mesh = mesh_analog.analog()
    degree = request.param
    vfspace = VectorFunctionSpace(mesh, family, degree)
    return fd2mm.FunctionSpaceAnalog(cl_ctx, mesh_analog, vfspace)


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

    fd2mm2fd_err = np.max(np.abs(original_function.dat.data - new_function.dat.data))
    assert fd2mm2fd_err < TOL, "Firedrake->Meshmode->Firedrake is not idempotent"

    mm2fd2mm_err = np.max(np.abs(original_field - new_field))
    assert mm2fd2mm_err < TOL, "Meshmode->Firedrake->Meshmode is not idempotent"


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


def test_identity_conversion(vector_function_space_analog):
    vfspace = vector_function_space_analog.analog()
    xx = SpatialCoordinate(vfspace.mesh())
    identity_fntn = Function(vfspace).interpolate(xx)

    identity_fntn_analog = fd2mm.FunctionAnalog(identity_fntn,
                                                vector_function_space_analog)

    identity_field = vector_function_space_analog.discretization().nodes()
    identity_field = identity_field.get(queue=queue)

    diff = np.max(np.abs(identity_fntn_analog.as_field() - identity_field))
    assert diff < TOL, "fd->mm identity conversion failed: " \
        "%f >= %f" % (diff, TOL)

    identity_copy = identity_fntn.copy(deepcopy=True)
    identity_fntn_analog.set_from_field(identity_field)
    diff = np.max(np.abs(identity_copy.dat.data
                         - identity_fntn_analog.analog().dat.data))
    assert diff < TOL, "mm->fd identity converesion failed: " \
        "%f >= %f" % (diff, TOL)
