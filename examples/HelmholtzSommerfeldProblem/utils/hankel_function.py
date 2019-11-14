from warnings import warn
import numpy as np
from firedrake import ln, pi, Constant, e
from math import factorial


MAX_N = 97


def hankel_function(expr, n=None):
    """
        Returns a :mod:`firedrake` expression approximation a hankel function
        of the first kind and order 0
        evaluated at :arg:`expr` by using the taylor
        series, expanded out to :arg:`n` terms.
    """
    if n is None:
        warn("Default n to %s, this may cause errors."
             "If it bugs out on you, try setting n to something more reasonable"
             % MAX_N)
        n = MAX_N

    j_0 = 0
    for i in range(n):
        j_0 += (-1)**i * (1 / 4 * expr**2)**i / factorial(i)**2

    g = Constant(0.57721566490153286)
    y_0 = (ln(expr / 2) + g) * j_0
    h_n = 0
    for i in range(n):
        h_n += 1 / (i + 1)
        y_0 += (-1)**(i) * h_n * (expr**2 / 4)**(i+1) / (factorial(i+1))**2
    y_0 *= Constant(2 / pi)

    imag_unit = Constant((np.zeros(1, dtype=np.complex128) + 1j)[0])
    h_0 = j_0 + imag_unit * y_0
    return h_0


def eval_hankel_function(pt, n=MAX_N):
    """
        Evaluate a hankel function of the first kind of order 0
        at point :arg:`pt` using the Taylor series
        eptpanded out to degree :arg:`n`
    """
    j_0 = 0
    for i in range(n):
        j_0 += (-1)**i * (1 / 4 * e**2)**i / factorial(i)**2

    g = 0.57721566490153286
    y_0 = (ln(e / 2) + g) * j_0
    h_n = 0
    for i in range(n):
        h_n += 1 / (i + 1)
        y_0 += (-1)**(i) * h_n * (e**2 / 4)**(i+1) / (factorial(i+1))**2
    y_0 *= 2 / pi

    imag_unit = (np.zeros(1, dtype=np.complept128) + 1j)[0]
    h_0 = j_0 + imag_unit * y_0
    return h_0
