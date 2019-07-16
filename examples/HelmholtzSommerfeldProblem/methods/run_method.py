from math import log, ceil
from firedrake import FunctionSpace, VectorFunctionSpace, Function, grad, \
    TensorFunctionSpace
from firedrake_to_pytential.op import FunctionConverter

from .pml import pml
from .nonlocal_integral_eq import nonlocal_integral_eq
from .transmission import transmission


trial_options = set(['mesh', 'degree', 'true_sol_expr'])

method_required_options = {'pml': set(['inner_region',
                                       'pml_x_region',
                                       'pml_y_region',
                                       'pml_xy_region',
                                       'pml_x_min',
                                       'pml_x_max',
                                       'pml_y_min',
                                       'pml_y_max']),
                           'nonlocal_integral_eq': set(['cl_ctx',
                                                        'queue']),
                           'transmission': set([])}

# All have the options arguments 'options_prefix' and
# 'solver_parameters'
method_options = {'pml': ['pml_type',
                          'delta',
                          'quad_const',
                          'speed'],
                  'nonlocal_integral_eq': ['fmm_order',
                                           'with_refinement',
                                           'qbx_order',
                                           'fine_order',
                                           ],
                  'transmission': []}


prepared_trials = {}


def trial_to_tuple(trial):
    return (trial['mesh'], trial['degree'], trial['true_sol_expr'])


def prepare_trial(trial, true_sol_name):
    tuple_trial = trial_to_tuple(trial)
    if tuple_trial not in prepared_trials:

        mesh = trial['mesh']
        degree = trial['degree']

        function_space = FunctionSpace(mesh, 'CG', degree)
        vect_function_space = VectorFunctionSpace(mesh, 'CG', degree)

        true_sol_expr = trial['true_sol_expr']
        true_solution = Function(function_space, name=true_sol_name).interpolate(
            true_sol_expr)
        true_solution_grad = Function(vect_function_space).interpolate(
            grad(true_sol_expr))

        prepared_trials[tuple_trial] = (mesh, function_space, vect_function_space,
                                        true_solution, true_solution_grad)

    return prepared_trials[tuple_trial]


memoized_objects = {}


def run_method(trial, method, wave_number,
               true_sol_name="True Solution",
               comp_sol_name="Computed Solution", **kwargs):
    """
        Returns (true solution, computed solution)

        :arg trial: A dict mapping each trial option to a valid value
        :arg method: A valid method (see the keys of *method_options*)
        :arg wave_number: The wave number

        kwargs should include the boundary id of the scatterer as 'scatterer_bdy_id'
        and the boundary id of the outer boundary as 'outer_bdy_id'

        kwargs should include the method options for :arg:`trial['method']`.
        for the given method.
    """
    # Get boundary ids
    scatterer_bdy_id = kwargs['scatterer_bdy_id']
    outer_bdy_id = kwargs['outer_bdy_id']

    # Get degree
    degree = trial['degree']

    # Get options prefix and solver parameters, if any
    options_prefix = kwargs.get('options_prefix', None)
    solver_parameters = kwargs.get('solver_parameters', None)

    # Get prepared trial args in kwargs
    prepared_trial = prepare_trial(trial, true_sol_name)
    mesh, fspace, vfspace, true_sol, true_sol_grad = prepared_trial

    # Create a place to memoize any objects if necessary
    tuple_trial = trial_to_tuple(trial)
    if tuple_trial not in memoized_objects:
        memoized_objects[tuple_trial] = {}

    comp_sol = None

    # Handle any special kwargs and get computed solution
    if method == 'pml':
        # Get required objects
        inner_region = kwargs['inner_region']
        pml_x_region = kwargs['pml_x_region']
        pml_y_region = kwargs['pml_y_region']
        pml_xy_region = kwargs['pml_xy_region']
        pml_x_max = kwargs['pml_x_max']
        pml_y_max = kwargs['pml_y_max']
        pml_x_min = kwargs['pml_x_min']
        pml_y_min = kwargs['pml_y_min']

        # Get optional argumetns
        pml_type = kwargs.get('pml_type', None)
        delta = kwargs.get('delta', None)
        quad_const = kwargs.get('quad_const', None)
        speed = kwargs.get('speed', None)

        # Make tensor function space
        if 'tfspace' not in memoized_objects[tuple_trial]:
            memoized_objects[tuple_trial]['tfspace'] = \
                TensorFunctionSpace(mesh, 'CG', degree)

        tfspace = memoized_objects[tuple_trial]['tfspace']

        ksp, comp_sol = pml(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                            options_prefix=options_prefix,
                            solver_parameters=solver_parameters,
                            inner_region=inner_region,
                            pml_x_region=pml_x_region,
                            pml_y_region=pml_y_region,
                            pml_xy_region=pml_xy_region,
                            fspace=fspace, tfspace=tfspace,
                            true_sol_grad=true_sol_grad,
                            pml_type=pml_type, delta=delta, quad_const=quad_const,
                            speed=speed,
                            pml_x_min=pml_x_min,
                            pml_y_min=pml_y_min,
                            pml_x_max=pml_x_max,
                            pml_y_max=pml_y_max)

    elif method == 'nonlocal_integral_eq':
        # Get required arguments
        cl_ctx = kwargs['cl_ctx']
        queue = kwargs['queue']

        # Set defaults for function converter
        with_refinement = kwargs.get('with_refinement', True)
        qbx_order = kwargs.get('qbx_order', degree)
        fine_order = kwargs.get('fine_order', 4 * degree)
        fmm_order = kwargs.get('fmm_order', 6)

        # }}}

        # Make function converter if not already built
        if 'function_converter' not in memoized_objects[tuple_trial]:
            function_converter = FunctionConverter(cl_ctx,
                                                   fine_order=fine_order,
                                                   fmm_order=fmm_order,
                                                   qbx_order=qbx_order,
                                                   with_refinement=with_refinement,
                                                   convert_only_near_bdy=True)
            memoized_objects[tuple_trial]['function_converter'] = function_converter

        function_converter = memoized_objects[tuple_trial]['function_converter']

        ksp, comp_sol = nonlocal_integral_eq(mesh, scatterer_bdy_id, outer_bdy_id,
                                             wave_number,
                                             options_prefix=options_prefix,
                                             solver_parameters=solver_parameters,
                                             fspace=fspace, vfspace=vfspace,
                                             true_sol=true_sol,
                                             true_sol_grad=true_sol_grad,
                                             cl_ctx=cl_ctx, queue=queue,
                                             function_converter=function_converter)

    elif method == 'transmission':
        ksp, comp_sol = transmission(mesh, scatterer_bdy_id, outer_bdy_id,
                                     wave_number,
                                     options_prefix=options_prefix,
                                     solver_parameters=solver_parameters,
                                     fspace=fspace,
                                     true_sol_grad=true_sol_grad)
    else:
        raise ValueError("Invalid method")

    comp_sol.rename(name=comp_sol_name)
    return true_sol, comp_sol, ksp
