"""
ADAPTED FROM pyamg-examples/ComplexSymmetric/demo_two_D_helmholtz.py

2D Helmholz Problem preconditioner
"""
import numpy as np
import scipy
import pyamg
import firedrake as fd

from .smoothed_aggregation_helmholtz_solver import \
    smoothed_aggregation_helmholtz_solver, planewaves, planewaves3D

from .my_vis import my_vis, shrink_elmts


class AMGTransmissionPreconditioner:
    def __init__(self, kappa, fspace, fd_mat, use_plane_waves=False,
                 tol=None, maxiter=None):
        #FIXME
        three_D = False
        if fspace.mesh().geometric_dimension() == 3:
            three_D = True
        elif fspace.mesh().geometric_dimension() != 2:
            raise ValueError("Must use geometric dimension 2 or 3 for pyamg, not %s" % 
                             fspace.mesh().geometric_dimension())
            

        # Set defaults for tol, maxiter
        if tol is None:
            tol = 1e-8
        if maxiter is None:
            maxiter = 20

        # Retrieve 2-D Helmholtz Operator and problem data.
        # This is operator was discretized using *fspace*
        A = fd_mat
        ai, aj, av = A.getValuesCSR()
        Asp = scipy.sparse.csr_matrix((av, aj, ai))

        # random initial guess for zero right-hand-side
        np.random.seed(625)
        self.x0 = np.random.rand(Asp.shape[0])

        omega = kappa
        vertices = np.zeros((fspace.mesh().coordinates.dat.data.shape[0], 3))
        if not three_D:
            vertices[:, :2] = np.real(fspace.mesh().coordinates.dat.data)
        else:
            vertices[:, :] = np.real(fspace.mesh().coordinates.dat.data)

        # Strength -- For level 0, aggregate based on distance so that only algebraic
        # neighbors at the same spatial location are aggregated together.  For all
        # coarse levels, all algebraic connections are considered strong.
        strength = [('distance', {'V': vertices, 'theta': 1e-5, 'relative_drop': False}),
                    ('symmetric', {'theta': 0.00})]

        # Prolongator smoother
        smooth = ('energy', {'krylov': 'cgnr'})

        # Aggregation -- non-standard 'naive' aggregation is done on level 0 so that
        # only algebraic neighbors at the same spatial location are aggregated
        # together.
        aggregate = ['naive', 'standard']

        # Note the matrix is complex-symmetric, not Hermitian, i.e. symmetry =
        # 'symmetric'.
        SA_build_args = {
            'max_levels': 10,
            'max_coarse': 50,
            'coarse_solver': 'pinv2',
            'symmetry': 'symmetric'}
        self.SA_solve_args = {
            'cycle': 'W',
            'maxiter': maxiter,
            'tol': tol,
            'accel': None}
        #   'accel': 'gmres'}

        # Pre- and post-smoothers -- gauss_seidel is an acceptable relaxation method
        # for resolved Helmholtz problems
        smoother = [('gauss_seidel', {'iterations': 4, 'sweep': 'forward'}),
                    ('gauss_seidel_nr', {'iterations': 4, 'sweep': 'forward'})]

        # improve_candidates[k] -- stipulates the relaxation method on level k
        # used to "improve" B
        improve_candidates = [
            ('gauss_seidel', {'iterations': 2, 'sweep': 'forward'}),
            ('gauss_seidel_nr', {'iterations': 1, 'sweep': 'forward'})]

        # Add plane waves if requested
        if use_plane_waves:
            # Now run a solver that introduces planewaves
            # Setup planewave parameters, such that planewaves[k] defines which planewaves
            # to introduce at level[k].  The final entry of None stipulates to introduce no
            # new planewaves from that level on down.
            X = vertices[:, 0].copy()
            Y = vertices[:, 1].copy()
            if three_D:
                Z = vertices[:, 2].copy()
                pwave_args = [None,
                              (planewaves3D, {'X': X, 'Y': Y, 'Z': Z, 'omega': omega,
                                            'angles': list(np.linspace(0., np.pi / 2., 2))}),
                              (planewaves3D, {'X': X, 'Y': Y, 'Z': Z, 'omega': omega,
                                            'angles': list(np.linspace(-np.pi / 8., 5 * np.pi / 8., 4))}),
                              None]
            else:
                pwave_args = [None,
                              (planewaves, {'X': X, 'Y': Y, 'omega': omega,
                                            'angles': list(np.linspace(0., np.pi / 2., 2))}),
                              (planewaves, {'X': X, 'Y': Y, 'omega': omega,
                                            'angles': list(np.linspace(-np.pi / 8., 5 * np.pi / 8., 4))}),
                              None]

            ##
            # Use constant in B for interpolation, but only between levels 0 and 1
            use_constant = (True, {'last_level': 0})

        # Otherwise Construct solver using the "naive" constant mode for B
        else:
            # Use constant in B for interpolation at all levels
            use_constant = (True, {'last_level': 10})
            pwave_args = [None]

        self.sa = smoothed_aggregation_helmholtz_solver(Asp,
                                                        planewaves=pwave_args,
                                                        use_constant=use_constant,
                                                        strength=strength,
                                                        smooth=smooth,
                                                        aggregate=aggregate,
                                                        improve_candidates=improve_candidates,
                                                        presmoother=smoother,
                                                        postsmoother=smoother,
                                                        **SA_build_args)
    def apply(self, pc, x, y):
        residuals = []
        # y <- A^{-1} x
        y[:] = self.sa.solve(x.getArray(), x0=self.x0, residuals=residuals, **self.SA_solve_args)[:]
        print("RESIDUALS=", residuals)
