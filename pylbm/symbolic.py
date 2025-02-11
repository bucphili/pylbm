# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Symbolic module
"""

import sys
import inspect
import numpy as np
import sympy as sp
from sympy.matrices.common import ShapeError

# List of symbols used in pylbm
nx, ny, nz, nv = sp.symbols("nx, ny, nz, nv", integer=True) #pylint: disable=invalid-name
ix, iy, iz, iv = sp.symbols("ix, iy, iz, iv", integer=True) #pylint: disable=invalid-name
ix_, iy_, iz_, iv_ = sp.symbols("ix_, iy_, iz_, iv_", integer=True) #pylint: disable=invalid-name
rel_ux, rel_uy, rel_uz = sp.symbols('rel_ux, rel_uy, rel_uz', real=True) #pylint: disable=invalid-name

class SymbolicVector(sp.Matrix):

    def __add__(self, other):
        if np.isscalar(other):
            mat = [a + other for a in self]
            return SymbolicVector._new(self.rows, self.cols, mat, copy=False)
        elif isinstance(other, (np.ndarray, SymbolicVector)):
            if self.shape[0] != other.shape[0]:
                raise ShapeError("SymbolicVector size mismatch: %s + %s." % (self.shape, other.shape))
            if len(other.shape) > 1:
                for s in other.shape[1:]:
                    if s != 1:
                        raise ShapeError("SymbolicVector size mismatch: %s + %s." % (self.shape, other.shape))
            mat = [a + b for a,b in zip(self, other)]
            return SymbolicVector._new(self.rows, self.cols, mat, copy=False)
        else:
            return super(SymbolicVector, self).__add__(other)

    def __radd__(self, other):
        return self + other

    def _multiply(self, other, method):
        if np.isscalar(other):
            mat = [a*other for a in self]
            return SymbolicVector._new(self.rows, self.cols, mat, copy=False)
        elif isinstance(other, (np.ndarray, SymbolicVector, sp.MatrixSymbol)):
            if self.shape[0] != other.shape[0]:
                raise ShapeError("SymbolicVector size mismatch: %s * %s." % (self.shape, other.shape))
            if len(other.shape) > 1:
                for s in other.shape[1:]:
                    if s != 1:
                        raise ShapeError("SymbolicVector size mismatch: %s * %s." % (self.shape, other.shape))
            mat = [a*b for a,b in zip(self, other)]
            return SymbolicVector._new(self.rows, self.cols, mat, copy=False)
        else:
            return getattr(super(SymbolicVector, self), method)(other)

    def __mul__(self, other):
        return self._multiply(other, '__mul__')

    def __rmul__(self, other):
        return self._multiply(other, '__rmul__')

def set_order(array, priority=None):
    """
    Reorder an array according to priority (lower to greater).

    Parameters
    ----------

    array : list
        array to reorder

    priority : list
        define how to reorder (lower to greater)
        (default is None)

    Return
    ------

    list
        new reordered list
    """
    if priority:
        out = []
        for p in np.argsort(priority):
            out.append(array[p])
        return out
    else:
        return array


def indexed(name, shape, index=[iv, ix, iy, iz], velocities=None,
            velocities_index=None, priority=None):
    """
    Return a SymPy matrix or an expression of indexed
    objects.

    Parameters
    ----------

    name : str
        name of the SymPy IndexedBase

    shape : list
        shape of the SymPy IndexedBase

    index : list
        indices of the indexed object
        (default is [iv, ix, iy, iz])

    velocities : list
        apply this list of integers on the space indices
        (default is None)

    velocities_index : list
        list of velocities used in the indexed output
        (default is None)

    priority : list
        define how to reorder the indeices (lower to greater)
        (default is None)

    Return
    ------

    sympy.Matrix or sympy.IndexedBase

    Examples
    --------

    >>> import sympy as sp
    >>> i, j, k = sp.symbols('i, j, k')
    >>> m = indexed("m", [10, 100, 200], [i, j, k])
    >>> m
    m[i, j, k]
    >>> m.shape
    (10, 100, 200)

    >>> m = indexed("m", [10, 100, 200], [i, j, k], velocities_index=range(4))
    >>> m
    Matrix([
    [m[0, j, k]],
    [m[1, j, k]],
    [m[2, j, k]],
    [m[3, j, k]]])

    >>> m = indexed("m", [10, 100, 200], [i, j, k], velocities_index=range(4), priority=[1, 2, 0])
    >>> m
    Matrix([
    [m[k, 0, j]],
    [m[k, 1, j]],
    [m[k, 2, j]],
    [m[k, 3, j]]])

    >>> m = indexed("m", [10, 100, 200], [i, j, k], velocities=[[0, 0], [1, 0], [-1, -1]])
    >>> m
    Matrix([
    [        m[0, j, k]],
    [    m[1, j + 1, k]],
    [m[2, j - 1, k - 1]]])

    """
    if velocities_index and velocities:
        raise ValueError("velocities and velocities_index can't be defined together.")

    output = sp.IndexedBase(name, set_order(shape, priority))

    if velocities_index:
        ind = [set_order([k] + list(index[1:]), priority) for k in velocities_index]
        return SymbolicVector([output[i] for i in ind])
    elif velocities is not None:
        ind = []
        indices = index[1:]
        for iv, v in enumerate(velocities): #pylint: disable=invalid-name
            tmp_ind = []
            for ik, k in enumerate(v): #pylint: disable=invalid-name
                tmp_ind.append(indices[ik] + int(k))
            ind.append(set_order([iv] + tmp_ind, priority))
        return SymbolicVector([output[i] for i in ind])
    else:
        return output[set_order(index, priority)]


def space_idx(ranges, priority=None):
    """
    Return a list of SymPy Idx with the bounds of the ranges.
    This list can be permuted if priority is defined.

    Parameters
    ----------

    ranges : list
        bounds of the range for each created SymPy Idx

    priority : list
        define how to reorder the ranges (lower to greater)
            (default is None)

    Return
    ------

    list
        list of SymPy Idx with the right ranges ordered by priority

    Examples
    --------

    >>> loop = space_idx([(0, 10), (-10, 10)])
    >>> loop
    [ix_, iy_]
    >>> loop[0].lower, loop[0].upper
    (0, 10)
    >>> loop = space_idx([(0, 10), (-10, 10)], priority=[1, 0])
    >>> loop
    [iy_, ix_]
    >>> loop[0].lower, loop[0].upper
    (-10, 10)

    """
    indices = [ix_, iy_, iz_]

    idx = []
    for ir, r in enumerate(ranges): #pylint: disable=invalid-name
        idx.append(sp.Idx(indices[ir], r))

    if priority:
        return set_order(idx, priority)
    else:
        return idx

def alltogether(M, nsimplify=False):
    """
    Simplify all the elements of sympy matrix M

    Parameters
    ----------

    M : sympy matrix
       matrix to simplify

    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if nsimplify:
                M[i, j] = M[i, j].expand().together().factor().nsimplify()
            else:
                M[i, j] = M[i, j].expand().together().factor()

def recursive_sub(expr, replace):
    for _ in range(len(replace)):
        new_expr = expr.subs(replace)
        if new_expr == expr:
            return new_expr
        else:
            expr = new_expr
    return new_expr

def call_genfunction(function, args):
    from .monitoring import monitor
    from .context import queue
    try:
        func_args = function.arg_dict.keys()
        d = {k:args[k] for k in func_args} #pylint: disable=invalid-name
        d['queue'] = queue
    except: #pylint: disable=bare-except
        func_args = inspect.getfullargspec(function).args
        d = {k:args[k] for k in func_args} #pylint: disable=invalid-name
    monitor(function)(**d)
