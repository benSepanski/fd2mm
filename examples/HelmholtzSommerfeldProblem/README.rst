The following is instructions for run_trial.py.
Most of it also pertains to multiprocessing_run_trial.py,
except for a few noted exceptions.

Specifying Trials to Run
========================

The file will run all possible combinations of trials produced from

* meshes in the file :code:`mesh_file_dir` which satisfy
  :code:`min_h <= h <= max_h`.
  The naming convention for meshes in the directory is :code:`max<h>.msh` with
  :code:`.` replaced by :code:`%`, e.g.
  if :code:`h=0.25`, then the file would be :code:`max0%25.msh` in
  :code:`mesh_file_dir`.

* :code:`kappa_list`

* :code:`degree_list`

* :code:`method_list`

i.e.

.. code-block:: python

    for each mesh
        for each degree
            for each kappa
                for each method
                    # run the given trial

Make sure that you set :code:`mesh_dim` to the geometric dimension of
the meshes in :code:`mesh_file_dir`


Setting Parameters
==================

This is primarily done by the dictionary :code:`method_to_kwargs`. For
each method, you can set various settings (to see all available
options, look at :code:`methods/run_method.py`). These settings
apply to all trials run.

Solver Parameters
-----------------

For each method you can set its own :code:`solver_parameters` (or 
you can use the command line, by prefixing with the method's
:code:`options_prefix`, which can be set in the :code:`method_to_kwargs` dict).

To record extreme singular values

1. Set at least one of 

    * :code:`-ksp_monitor_singular_value`
    * :code:`-ksp_compute_singularvalues`
    * :code:`-ksp_compute_eigenvalues`

2. Set :code:`-ksp_gmres_restart` to some high value

There are two special parameters which are not the typical
petsc options

1. :code:`'gamma'` a complex parameter :math:`\gamma`, defaults to 1.0
2. :code:`'beta'`, a complex parameter :math:`\beta`, defaults to :math:`\sqrt{\gamma}`

Transmission and the nonlocal coupling are preconditioned by

.. math::

        \begin{cases}
        (-\Delta - \kappa^2 \gamma) u(x) = 0 & x \in \Omega \\
        (\frac{\partial}{\partial n} - i\kappa\beta)u(x) = 0 & x \in \Sigma
        \end{cases}


Other Options
=============

* Set :code:`use_cache = True` to use previously computed results (e.g.
  if you just want to print the error). Regardless, results
  are stored in a .csv in `data/` corresponding to the mesh
  directory name.
* Set :code:`write_over_duplicate_trials` over :code:`True` if you want to
  write over already-computed trials (i.e. you are re-computing them,
  so :code:`use_cache` is :code:`False`).
* In 2d, set :code:`visualize` to :code:`True` if you want each solution
  to be plotted.
* :code:`get_fmm_order(kappa, h)` returns the fmm order you want
  pytential to use given kappa and h. Pytential guarantees
  accuracy of :math:`||\text{err}||_\infty \leq c^(p+1)`,
  where :math:`c` is 0.5 in 2d and 0.75 in 3d, and :math:`p` is
  the fmm order.

Multiprocessing Options
-----------------------

In :code:`multiprocessing_run_trials.py` the trials are run in parallel.
You now have the options

* :code:`num_processes`
* :code:`print_trials` (print trial results as computed)
