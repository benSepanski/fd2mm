from math import log, ceil
from firedrake import FunctionSpace, VectorFunctionSpace, Function, grad, \
    TensorFunctionSpace
from firedrake_to_pytential.op import FunctionConverter

from methods.pml import pml
from methods.nonlocal_integral_eq import nonlocal_integral_eq
from methods.transmission import transmission


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

method_options = {'pml': ['pml_type',
                          'delta',
                          'quad_const',
                          'speed'],
                  'nonlocal_integral_eq': ['epsilon',
                                           'with_refinement',
                                           'qbx_order',
                                           'fine_order'],
                  'transmission': []}


prepared_trials = {}


def trial_to_tuple(trial):
    return (trial['mesh'], trial['degree'], trial['true_sol_expr'])


def prepare_trial(trial):
    tuple_trial = trial_to_tuple(trial)
    if tuple_trial not in prepared_trials:

        mesh = trial['mesh']
        degree = trial['degree']

        function_space = FunctionSpace(mesh, 'CG', degree)
        vect_function_space = VectorFunctionSpace(mesh, 'CG', degree)

        true_sol_expr = trial['true_sol_expr']
        true_solution = Function(function_space).interpolate(true_sol_expr)
        true_solution_grad = Function(vect_function_space).interpolate(
            grad(true_sol_expr))

        prepared_trials[tuple_trial] = (mesh, function_space, vect_function_space,
                                        true_solution, true_solution_grad)

    return prepared_trials[tuple_trial]


memoized_objects = {}


def run_trial(trial, method, wave_number, **kwargs):
    """
        Returns the computed solution

        :arg trial: A dict mapping each trial option to a valid value
        :arg method: A valid method (see the keys of *method_options*)
        :arg wave_number: The wave number

        kwargs should include the boundary id of the scatterer as 'scatterer_bdy_id'
        and the boundary id of the outer boundary as 'outer_bdy_id'

        kwargs should include the method options for :arg:`trial['method']`.
        These will be used under the hood to set the method_kwargs
        for the given method.

        Also, kwargs can have any of the following which may have
        already been computed

    """
    assert 'scatterer_bdy_id' in kwargs
    assert 'outer_bdy_id' in kwargs

    method_kwargs = dict(kwargs)  # copy all of the kwargs given
    degree = trial['degree']
    method_kwargs['degree'] = degree

    prepared_trial = prepare_trial(trial)
    mesh, fspace, vfspace, true_sol, true_sol_grad = prepared_trial
    # Get prepared trial args in kwargs
    method_kwargs['mesh'] = mesh
    method_kwargs['fspace'] = fspace
    method_kwargs['vfspace'] = vfspace
    method_kwargs['true_sol'] = true_sol
    method_kwargs['true_sol_grad'] = true_sol_grad

    tuple_trial = trial_to_tuple(trial)
    if tuple_trial not in memoized_objects:
        memoized_objects[tuple_trial] = {}

    # Handle any specialty kwargs and defaults, then get the function
    method_fntn = None
    if method == 'pml':
        method_fntn = pml

        # Set defaults
        method_kwargs['pml_type'] = method_kwargs.get('pml_type', 'bdy_integral')
        method_kwargs['delta'] = method_kwargs.get('delta', 0.1)
        method_kwargs['quad_const'] = method_kwargs.get('quad_const', 1.0)
        method_kwargs['speed'] = method_kwargs.get('speed', 340)

        # Make tensor function space
        if 'tfspace' not in memoized_objects[tuple_trial]:
            memoized_objects[tuple_trial]['tfspace'] = \
                TensorFunctionSpace(mesh, 'CG', degree)

        method_kwargs['tfspace'] = memoized_objects[tuple_trial]['tfspace']

    elif method == 'nonlocal_integral_eq':
        method_fntn = nonlocal_integral_eq

        # Set defaults
        with_refinement = kwargs.get('with_refinement', True)
        qbx_order = kwargs.get('qbx_order', degree)
        fine_order = kwargs.get('fine_order', 4 * degree)

        # {{{ Compute fmm order
        epsilon = method_kwargs.get('epsilon', 0.05)
        epsilon = min(epsilon, 1.0)  # No reason to have this bigger than 1
        if 'epsilon' in method_kwargs:
            del method_kwargs['epsilon']  # Will be replaced by *fmm_order*

        if mesh.geometric_dimension() == 2:
            base = 0.5
        elif mesh.geometric_dimension() == 3:
            base = 0.75
        else:
            raise ValueError("Ambient dimension must be 2 or 3")

        # If fmm_order is p and base is b, p should be so that
        r"""

        ..math ::

            \epsilon <= b^(p + 1)

        """
        # Do the above, but make sure it's at least one.
        fmm_order = max(ceil(log(epsilon, base) - 1), 1)
        # }}}

        # Set defaults
        method_kwargs['with_refinement'] = with_refinement
        method_kwargs['qbx_order'] = qbx_order
        method_kwargs['fine_order'] = fine_order
        method_kwargs['fmm_order'] = fmm_order

        # Make function converter if not already built
        if 'function_converter' not in memoized_objects[tuple_trial]:
            function_converter = FunctionConverter(method_kwargs['cl_ctx'],
                                                   fine_order=fine_order,
                                                   fmm_order=fmm_order,
                                                   qbx_order=qbx_order,
                                                   with_refinement=with_refinement)
            memoized_objects[tuple_trial]['function_converter'] = function_converter

        method_kwargs['function_converter'] = \
            memoized_objects[tuple_trial]['function_converter']

    elif method == 'transmission':
        method_fntn = transmission

    else:
        raise ValueError("Invalid method")

    # Make sure required args are present
    assert method_required_options[method] <= method_kwargs.keys()

    return true_sol, method_fntn(wave_number, **method_kwargs)
