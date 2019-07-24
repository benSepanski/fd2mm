The following is instructions for run_trial.py.
Most of it also pertains to multiprocessing_run_trial.py,
except for a few noted exceptions.

Specifying Trials to Run
========================

The file will run all possible combinations of trials produced from

* meshes in the file `mesh_file_dir` which satisfy `min_h <= h <= max_h`.
  The naming convention for meshes in the directory is `max<h>.msh` with
  `.` replaced by `%`, e.g.
  if `h=0.25`, then the file would be `max0%25.msh` in `mesh_file_dir`.

* `kappa_list`

* `degree_list`

* `method_list`

i.e.

.. code-block:: python

    for each mesh
        for each degree
            for each kappa
                for each method
                    # run the given trial

Make sure that you set `mesh_dim` to the geometric dimension of
the meshes in `mesh_file_dir`


Setting Parameters
==================

This is primarily done by the dictionary `method_to_kwargs`. For
each method, you can set various settings (to see all available
options, look at `methods/run_method.py`). These settings
apply to all trials run.

Solver Parameters
-----------------

For each method you can set its own `solver_parameters` (or 
you can use the command line, by prefixing with the method's
`options_prefix`, which can be set in the `method_to_kwargs` dict).

There are two special parameters which are not the typical
petsc options

1. `'gamma'` a complex parameter :math:`\gamma`, defaults to 1.0
2. `'beta'`, a complex parameter :math:`\beta`, defaults to 1.0

Transmission and the nonlocal coupling are preconditioned by

.. math::

        \begin{cases}
        (-\Delta - \kappa^2 \gamma) u(x) = 0 & x \in \Omega \\
        (\frac{\partial}{\partial n} - i\kappa\beta)u(x) = 0 & x \in \Sigma
        \end{cases}


Other Options
=============

* Set `use_cache = True` to use previously computed results (e.g.
  if you just want to print the error). Regardless, results
  are stored in a .csv in `data/` corresponding to the mesh
  directory name.
* Set `write_over_duplicate_trials` over `True` if you want to
  write over already-computed trials (i.e. you are re-computing them,
  so `use_cache` is `False`).
* In 2d, set `visualize` to `True` if you want each solution
  to be plotted.
* `get_fmm_order(kappa, h)` returns the fmm order you want
  pytential to use given kappa and h. Pytential guarantees
  accuracy of :math:`||\text{err}||_\infty \leq c^(p+1)`,
  where :math:`c` is 0.5 in 2d and 0.75 in 3d, and :math:`p` is
  the fmm order.

Multiprocessing Options
-----------------------

In `multiprocessing_run_trials.py` the trials are run in parallel.
You now have the options

* `num_processes`
* `print_trials` (print trial results as computed)
