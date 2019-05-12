import pickle
#from os.path import isfile
import os

import pyopencl as cl
import pyopencl.clmath

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

from firedrake import FunctionSpace, VectorFunctionSpace, \
    MeshHierarchy, Function, SpatialCoordinate, solve, \
    sqrt, assemble, dx, ds, Mesh, Constant, inner, grad, \
    TrialFunction, TestFunction, FacetNormal, plot
from firedrake_to_pytential.op import FunctionConverter
import numpy as np
from math import log
import matplotlib.pyplot as plt

from gmati_coupling import gmati_coupling
from nitsche import nitsche
from integral_eq_direct import integral_eq_direct
from green_to_dirichlet import green_to_dirichlet
from pml_driver import pml
from pml_functions import hankel_function

omega_list = [250]
c = 340
degree_list = [1]
fmm_orders = [20]
method_list = ['coupling']
hankel_expansion_cutoff = None
# If False, prints in a readable format. Else, prints to file_name
# in format consistent with Latex's tabular format
to_file = True
file_name = "gmati.tex"
file_dir = "tex_files/"  # make sure you have a "/"
use_pickle = False  # Use cached results
iterative_cache = True  # cache during computation
cache_file = "cached_results.pickle"
mesh_file_dir = "msh_files/"
with_refinement = True

try:
    pickle_in = open(file_dir + cache_file, "rb")
    known_results = pickle.load(pickle_in)
    pickle_in.close()
except (OSError, IOError):
    known_results = {}
prev_known_results = dict(known_results)

# Allowable methods:
known_methods = {'coupling', 'pml(bdyint)', 'pml(quadratic)', 'transmission',
                 'nitsche', 'integral_eq_direct', 'green_to_dirichlet'}
assert set(method_list) <= known_methods

# for the double layer formulation, the normal points
print("Reading meshes...")
meshes = []
for filename in os.listdir(mesh_file_dir):
    if filename.endswith('.msh'):
        meshes.append(Mesh(mesh_file_dir + filename))
meshes.sort(key=lambda x: x.coordinates.dat.data.shape[0])
print("Meshes read in")

pml_x_region = 1
pml_y_region = 2
pml_xy_region = 3
inner_region = 4
outer_bdy_id = 6
inner_bdy_id = 5


# Results are stored as a tuple of dictionaries
# (input_keys: input_values, output_keys: output_values)
input_keys = ('method', 'kappa', 'ndof', 'degree', 'fmm_order')
output_keys = ('rel_err',)

assert len(set(input_keys) & set(output_keys)) == 0

def result_to_kv(result):
    result_in = tuple([result[0][key] for key in input_keys])
    result_out = tuple([result[1][key] for key in output_keys])
    return (result_in, result_out)


# Here we have printable names, every input_key and output_key
# *MUST* have a printable name:
printable_names = {'method': 'Method', 'kappa': 'Kappa',
                   'ndof': 'd.o.f. Count', 'degree': 'Degree',
                   'fmm_order': 'FMM Order',
                   'rel_err': 'Relative Error',
                   'coupling': 'Boundary Integral Operators',
                   'nitsche': 'Nitsche',
                   'green_to_dirichlet': "Dirichlet BC from Green's",
                   'integral_eq_direct': 'Integral Equations Directly',
                   'pml(bdyint)': 'PML (Boundary Integral Absorbing)',
                   'pml(quadratic)': 'PML(Quadratic Absorbing)',
                   'transmission': 'Transmission'}
# Latex names default to printable names if unlisted
latex_names = {'kappa': '$\\kappa$'}

# rows sorted by row_sort[0], sub-sorted by row_sort[1], etc.
row_sort = ['kappa', 'degree', 'ndof', 'method', 'fmm_order']
# order of columns
column_order = ['kappa', 'degree', 'ndof', 'rel_err', 'method', 'fmm_order']
# Set up plotting (x, y, [(key, required value)], [key to put on same graph])
# Nb: keys to put on some graph must be input keys
plots = [('ndof', 'rel_err', [], ['method'])]


# Why pass key and value? Sometimes you want to format a value
# according to the key that it represents. Just given a key
# forms the key, if given a (k, v) pair it formats the key
# and formats the value according to the key
def printable_form(key, value=None):
    printable_key = printable_names[key]
    if value is None:
        return printable_key

    if value in printable_names:
        printable_value = printable_names[value]
    elif isinstance(value, float):
        if 1e-1 < value < 1e2:
            printable_value = "%.2f" % (round(value, 2))
        else:
            printable_value = "%.2e" % (value)
    else:
        printable_value = str(value)

    return (printable_key, printable_value)


def latex_form(key, value=None):
    if key in latex_names:
        printable_key = latex_names[key]
    else:
        printable_key = printable_names[key]
    if value is None:
        return printable_key

    if value in latex_names:
        printable_value = latex_names[value]
    else:
        _, printable_value = printable_form(key, value)

    return (printable_key, printable_value)


def run_method(method, V, kappa, true_sol_expr, quad_max=100, quad_step=10,
               function_converter=None):
    # ensure method is in known methods
    assert method in known_methods

    amg_params = {
        "ksp_type": "gmres",
        "ksp_monitor": True,
        "pc_type": "gamg"
    }

    direct_params = {
        "ksp_type": "preonly",
        "pc_type": "lu"
    }

    pml_params = direct_params
    trans_params = amg_params

    if method == 'coupling':
        return gmati_coupling(cl_ctx, queue, V, kappa,
                              outer_bdy_id, inner_bdy_id,
                              true_sol_expr, function_converter)
    elif method == 'nitsche':
        return nitsche(cl_ctx, queue, V, kappa,
                       outer_bdy_id, inner_bdy_id,
                       true_sol_expr, function_converter)
    elif method == 'green_to_dirichlet':
        return green_to_dirichlet(cl_ctx, queue, V, kappa,
                                  outer_bdy_id, inner_bdy_id,
                                  true_sol_expr, function_converter)
    elif method == 'integral_eq_direct':
        return integral_eq_direct(cl_ctx, queue, V, kappa,
                                  outer_bdy_id, inner_bdy_id,
                                  true_sol_expr, function_converter)
    elif method == 'pml(bdyint)':
        return pml('bdy_int', V, kappa, c,
                   true_sol_expr,
                   outer_bdy_id, inner_bdy_id, inner_region,
                   pml_x_region, pml_y_region,
                   pml_xy_region, delta=1e-6,
                   solver_params=pml_params)
    elif method == 'pml(quadratic)':
        return pml('quadratic', V, kappa, c,
                   true_sol_expr,
                   outer_bdy_id, inner_bdy_id, inner_region,
                   pml_x_region, pml_y_region,
                   pml_xy_region, delta=1e-6,
                   quad_max=quad_max,
                   quad_step=quad_step,
                   solver_params=pml_params)
    elif method == 'transmission':
        V_squared = VectorFunctionSpace(V.mesh(), V.ufl_element())
        grad_h = Function(V_squared).interpolate(grad(true_sol_expr))

        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(grad(u), grad(v)) * dx - kappa**2 * inner(u, v) * dx \
            - 1j * kappa * inner(u, v) * ds(outer_bdy_id)
        n = FacetNormal(V.mesh())
        L = inner(inner(grad_h, n), v) * ds(inner_bdy_id)
        p = Function(V)
        solve(a == L, p, solver_parameters=trans_params)

        return p

results = []
total_iters = \
    len(degree_list) * len(meshes) * len(omega_list) * len(method_list)
cur_iter = 0
for degree_nr, degree in enumerate(degree_list):
    for mesh_nr, m in enumerate(meshes):
        # Delay computation of this since results may already be cached
        V = FunctionSpace(m, 'CG', degree)
        ndof = V.dof_count
        xx = SpatialCoordinate(m)

        quad_min = 0.1
        quad_max = 10
        quad_step = 0.5
        fine_order = degree
        qbx_order = degree

        for fmm_order in fmm_orders:
            function_converter = FunctionConverter(cl_ctx,
                                                   fine_order=fine_order,
                                                   fmm_order=fmm_order,
                                                   qbx_order=qbx_order,
                                                   with_refinement=with_refinement)

            for kappa_nr, omega in enumerate(omega_list):
                kappa = omega / c
                # Set up true sol
                true_sol_expr = Constant(1j / 4) * \
                    hankel_function(kappa * sqrt((xx[0] - 0.5)**2 + xx[1]**2),
                                    n=hankel_expansion_cutoff)
                true_sol = Function(V, name="True Solution").interpolate(true_sol_expr)

                true_sol_norm = sqrt(assemble(inner(true_sol, true_sol) * dx(inner_region)))

                for method_nr, method in enumerate(method_list):
                    cur_iter += 1
                    print("Iteration %s/%s" % (cur_iter, total_iters))
                    print("degree=", degree)
                    print("ndof=", ndof)
                    print("method=", method)

                    result_input = {'method': method,
                        'kappa': kappa,
                        'ndof': ndof,
                        'degree': degree,
                        'fmm_order': fmm_order}
                    result_key, _ = result_to_kv([result_input, {'rel_err': None}])
                    # If using cached results, and have them stored, use them
                    if use_pickle and result_key in known_results:
                        result_output = known_results[result_key]
                        result_output = {k: v for (k, v) in zip(output_keys, result_output)}
                        results.append((result_input, result_output))
                        print()
                        continue

                    solution = run_method(method, V, kappa, true_sol_expr, quad_max, quad_step,
                                          function_converter)
                    error = Function(V).interpolate(true_sol - solution)
                    error_fntn = error
                    error = assemble(inner(error, error) * dx(inner_region))

                    result_output = {}
                    result_output['rel_err'] = error / true_sol_norm
                    results.append((result_input, result_output))
                    # store in known results
                    result_key, result_val = result_to_kv(results[-1])
                    known_results[result_key] = result_val
                    print("rel err=", result_output['rel_err'])
                    print("err=", error)
                    print()
                    #plot(error_fntn)
                    #plt.show()

            if iterative_cache:
                # store results in cache after each fmm order
                pickle_out = open(file_dir + cache_file, "wb")
                pickle.dump(known_results, pickle_out)
                pickle_out.close()

# store results in cache
pickle_out = open(file_dir + cache_file, "wb")
pickle.dump(known_results, pickle_out)
pickle_out.close()

# {{{ Output table to .tex file

out_file = open(file_dir + file_name, 'w')

out_file.write("\\documentclass[uft8]{article}\n")
out_file.write("\\usepackage{amsmath}\n")
out_file.write("\\usepackage{graphicx}\n")
out_file.write("\\usepackage{longtable}\n")

out_file.write("\n\\begin{document}\n")

# Remove columns if only 1 value
present_columns = set(column_order)

col_with_list = [('kappa', [om / c for om in omega_list]),
                 ('degree', degree_list),
                 ('method', method_list),
                 ('fmm_order', fmm_orders)]
for (column, column_list) in col_with_list:
    # Only columns that can be removed are inputs
    assert column in input_keys

    if len(column_list) == 1:
        key, val = latex_form(column, column_list[0])
        out_file.write(key + " = " + val + "\n\n")
        present_columns.remove(column)

# Sort rows of data
for ky in row_sort[::-1]:
    if ky in present_columns:
        results.sort(key=lambda x: x[0][ky])

out_file.write("\\begin{longtable}{")
for _ in range(len(present_columns) - 1):
    out_file.write("c ")
out_file.write("c}\n")

i = 0
for col in column_order:
    if col in present_columns:
        i += 1
        to_write = latex_form(col)
        if i < len(present_columns):
            out_file.write(to_write + " & ")
        else:
            out_file.write(to_write + " \\\\ \\hline\n")

for result in results:
    i = 0
    for col in column_order:
        if col in present_columns:
            if col in input_keys:
                _, value = latex_form(col, result[0][col])
            else:
                _, value = latex_form(col, result[1][col])

            i += 1
            if i < len(present_columns):
                out_file.write(value + " & ")
            else:
                out_file.write(value + " \\\\ \\hline\n")

out_file.write("\\end{longtable}\n\n")

# }}}


# {{{ Do plotting and put in .tex file

# Set up plotting (x, y, [(key, required value)], [key to put on same graph])
print("Plotting...")

for (x_key, y_key, required_vals, vals_on_graph) in plots:
    required_keys = set([r[0] for r in required_vals])
    keys_on_graph = set(vals_on_graph)
    free_keys = set(input_keys) & present_columns
    free_keys -= (required_keys | keys_on_graph | set([x_key, y_key]))

    assert keys_on_graph <= set(input_keys)

    x_key_ind, y_key_ind = 0, 0
    if x_key in output_keys:
        x_key_ind = 1
    if y_key in output_keys:
        y_key_ind = 1

    # get x and y
    x, y = {}, {}
    for result in results:
        valid = True
        for req_key, req_val in required_vals:
            i = 0
            if req_key in output_keys:
                i = 1
            if result[i][req_key] != req_val:
                valid = False
                break
        if valid:
            free_ndx = tuple([(ky, result[0][ky]) for ky in free_keys])
            x.setdefault(free_ndx, {})
            y.setdefault(free_ndx, {})

            on_graph_ndx = tuple([(ky, result[0][ky]) for ky in vals_on_graph])
            x[free_ndx].setdefault(on_graph_ndx, []).append(result[x_key_ind][x_key])
            y[free_ndx].setdefault(on_graph_ndx, []).append(result[y_key_ind][y_key])

    # For each free_ndx make a plot consisting of overlays of all the graph_ndx
    for free_ndx in x:
        plot_fname = "fig--free"
        for key, val in free_ndx:
            plot_fname += '_' + str(key) + '-' + str(val)
        plot_fname += "--ongraph"
        for key in keys_on_graph:
            plot_fname += "_" + str(key)
            key_vals = set()
            for result in results:
                key_vals.add(result[0][key])
            key_vals = sorted(list(key_vals))
            for val in key_vals:
                plot_fname += '-' + str(val)
        plot_fname += '--req'
        for req, req_val in required_vals:
            plot_fname += "_" + str(req) + '-' + str(req_val)
        plot_fname += '--x_' + str(x_key)
        plot_fname += '--y_' + str(y_key)

        plot_fname = plot_fname.replace('.', '$')  # so that reads appropriate extension
        plot_fname += ".png"

        make_plot = True
        """ USED FOR caching plots
        if (use_pickle and isfile(file_dir + plot_fname) and
                get_result_ndx(result) in prev_known_results):
            make_plot = False
        """

        if make_plot:
            fig, ax = plt.subplots()
            for graph_ndx in x[free_ndx]:
                label = ""
                for i, (col, arg) in enumerate(graph_ndx):
                    if len(graph_ndx) > 1:
                        label = latex_form(col) + " = "
                    label += latex_form(col, arg)[1]
                    if i < len(graph_ndx) - 1:
                        label += "; "
                xplot = x[free_ndx][graph_ndx]
                yplot = y[free_ndx][graph_ndx]
                if x_key == 'rel_err':
                    xplot = [log(entry) for entry in xplot]
                if y_key == 'rel_err':
                    yplot = [log(entry) for entry in yplot]
                ax.plot(xplot, yplot, '.-', label=label)
            ax.legend(loc='upper center')
            if x_key == 'rel_err':
                ax.set_xlabel('log(' + printable_form(x_key) + ')')
            else:
                ax.set_xlabel(printable_form(x_key))
            if y_key == 'rel_err':
                ax.set_ylabel('log(' + printable_form(y_key) + ')')
            else:
                ax.set_ylabel(printable_form(y_key))
            fig.savefig(file_dir + plot_fname)

        # include in pdf
        out_file.write("\\includegraphics[width=\\textwidth]{"+plot_fname+"}\\\\ \n")
        for key, val in free_ndx:
            key, val = latex_form(key, val)
            out_file.write(str(key) + " = " + str(val) + ";\n")
        out_file.write("\n\n")

print("Plotting done.")
# }}}

out_file.write("\\end{document}\n")
out_file.close()
