
# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sympy as sp
from sympy import Eq

def euler(lhs, rhs):
    """
    Explicit Euler in symbolic form.
    """
    #dt = sp.Symbol('dt')
    return Eq(lhs, lhs + 1/2*rhs,evaluate=False)
