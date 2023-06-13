#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.linalg import lapack

jax.config.update("jax_enable_x64", True)


class Color:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'


def is_psd(mat):
    return np.all(np.linalg.eigvals(mat) >= 0)


def is_pd(mat):
    return np.all(np.linalg.eigvals(mat) > 0)


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def mask_to_psd(Mat, ratio):
    """ change matrix's eigen value to zero

    :param M:
    :return:
    """
    assert Mat.shape[0] == Mat.shape[1]
    eigval, eigvec = np.linalg.eigh(Mat)
    assert np.all(eigval > 0), "mat is not pd"
    mask = np.random.rand(*eigval.shape)
    eigval[mask >= ratio] = 0.0
    res = eigvec @ np.diag(eigval) @ eigvec.T
    return res

def mask_to_Newton(Mat, a, b):
    """ 
    aI <= Mat <= bI

    :param Mat:
    :return:
    """
    assert Mat.shape[0] == Mat.shape[1]
    assert a < b , "B is larger than A"
    eigval, eigvec = np.linalg.eigh(Mat)
    # assert np.allclose(eigvec @ np.diag(eigval) @ eigvec.T, Mat),"Yaba"
    print("min eigval", np.min(eigval))
    print("max eigval", np.max(eigval))
    eigval[eigval <= a] = a
    eigval[eigval >= b] = b
    res = eigvec @ np.diag(eigval) @ eigvec.T
    return res

def mask_to_mat_lower(Mat, ratio, r):
    """ make min eigvals of Mat is -r

    :param Mat:
    :param ratio:
    :param r:
    :return:
    """
    assert r >= 0, "r is negative"
    assert Mat.shape[0] == Mat.shape[1]
    eigval, eigvec = np.linalg.eigh(Mat)
    assert np.all(eigval > 0), "mat is not pd"
    mask = np.random.rand(*eigval.shape)
    eigval[mask >= 1.0 - ratio] = -r
    # [1.0-0.0] > 0.8 is 20 % of elements
    res = eigvec @ np.diag(eigval) @ eigvec.T
    return res


def gen_random_pd(matrixSize):
    """ generate randomly generated symmetric positive definite matrix

    :param matrixSize: the size of matrixSize
    :return:
    """
    A = np.random.rand(matrixSize, matrixSize)
    B = A @ A.T
    assert is_symmetric(B), "not a symmetric matrix"
    return B


def jit_create_symmetric(vals):
    c = vals.shape[0]
    m = int((-1 + np.sqrt(1 + 8 * c)) // 2)
    assert m * (m + 1) == 2 * vals.shape[0], "shape mismatch inside sqrt is " + str(c * 8 + 1)
    res = jax.jit(lambda val: create_symmetric(val, m))
    return res(vals)


def create_symmetric(vals, m: int):
    """ create a symmetric matrix from a vector
    # https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
    :param vals: vectors
    :param m: size
    :return: matrix
    """
    assert m * (m + 1) == 2 * vals.shape[0], "shape mismatch"
    new = jnp.zeros((m, m))
    idx = jnp.triu_indices_from(new)
    new = jax.ops.index_update(new, jax.ops.index[idx], vals)
    new = jax.ops.index_update(new, jax.ops.index[(idx[1], idx[0])], vals)
    return new


def min_eigv(Mat):
    """ return minimum eigen value

    :param Mat:
    :return:
    """
    return sp.linalg.eigh(Mat, eigvals_only=True, subset_by_index=[0, 0])


@jax.jit
def sym2vec(mat):
    """ covnert symmetric matrix to vector

    :param vec:
    :return:
    """
    size = mat.shape[0]
    res = mat[0, :]
    for i in range(1, size):
        # print(res.shape, mat[i, i:size].shape)
        res = jnp.concatenate([res, mat[i, i:size]])
    assert res.shape[0] == (size + 1) * size // 2
    return res


def first_finite_differences(f, x, eps=1e-10):
    """
    https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
    :param f:
    :param x:
    :return:
    """
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2.0 * eps) for v in jnp.eye(len(x))])


def upper_triangular_to_symmetric(ut):
    """
    Destructive method
    https://stackoverflow.com/questions/58718365/fast-way-to-convert-upper-triangular-matrix-into-symmetric-matrix
    :param ut:
    :return:
    """
    n = ut.shape[0]
    inds = np.tri(n, k=-1, dtype=np.bool)
    ut[inds] = ut.T[inds]
    return None


@jax.jit
def jax_upper_triangular_to_symmetric(ut):
    """ Not Destructive slower

    :param ut:
    :return:
    """
    n = ut.shape[0]
    ilow = np.tril_indices(n, -1)
    ut = jax.ops.index_update(ut, jax.ops.index[ilow], ut.T[ilow])
    return ut


def pos_sym_inv(m):
    """
    https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
    :param m:
    :return:
    """
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        jnp.savetxt("outputs/dpotrf.csv", m, delimiter=",")
        raise ValueError('Info = {val} dpotrf failed on input {matrix}'.format(val=info, matrix=m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv


def jax_pos_sym_inv(a):
    n = a.shape[0]
    I = jnp.identity(n)
    return jax.scipy.linalg.solve(a, I, sym_pos=True, overwrite_b=True)

def modified_bfgs_update(Bk, sk, qk):
    """
    sk := x_{k+1} - x_k
    qk := \nabla psi_mu(x_{k+1}, y_{k+1}, z_{k+1}) - \nabla psi_mu(x_{k}, y_{k+1}, z_{k+1})
    """
    assert sk.shape == qk.shape
    assert Bk.shape == (sk.shape[0], sk.shape[0])
    sk = sk.reshape(sk.shape[0], 1)
    qk = qk.reshape(qk.shape[0], 1)
    Bksk = Bk @ sk
    skBksk = sk.T @ Bk @ sk
    skqk =  sk.T @ qk
    if skqk < 0.2 * skBksk:
        psi_k = 0.8 * skBksk /(skBksk - skqk)
        qkhat = psi_k * qk + (1.0-psi_k)* Bksk
    else:
        qkhat =  qk
    second =  (Bksk @ Bksk.T)/skBksk
    third = (qkhat @ qkhat.T )/(sk.T @ qkhat)
    Bkp = Bk - second + third
    Bkp = (Bkp + Bkp.T)/2.0
    # assert np.linalg.norm(((qkhat @ qkhat.T )/(sk.T @ qkhat))@sk - qkhat) < 1e-4
    # assert np.linalg.norm((Bk-(Bksk @ Bksk.T)/skBksk)@sk) <1e-4, str(np.linalg.norm((Bk-(Bksk @ Bksk.T)/skBksk)@sk))
    eigval, eigvec = np.linalg.eigh(Bkp)
    # assert np.allclose(eigvec @ np.diag(eigval) @ eigvec.T, Mat),"Yaba"
    print("min eigval", np.min(eigval))
    print("max eigval", np.max(eigval))
    if not is_pd(Bkp):
        return np.identity(Bkp.shape[0])
    if np.linalg.norm(Bkp @ sk - qkhat) > 1.0:
        return np.identity(Bkp.shape[0])
    return Bkp