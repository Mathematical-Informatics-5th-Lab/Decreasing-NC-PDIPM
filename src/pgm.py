#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Projected gradient method for
Positive semidefinite matrix factorization

 minimize sum_i sum_j (X_{ij} - trace(AiBj))^2
 subject to Ai+rI Bj+rI are positive semidefinite matrix
 A+ r I >= 0, B+rI>=0

The number of A is m
The number of B is n
The size of A and B is k times k

"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from src.psf import gen_psf_params, psf, gen_shifted_params
from src.utils import jit_create_symmetric, sym2vec, min_eigv

jax.config.update("jax_enable_x64", True)


def project_S_plus(A):
    """
    # https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix/63131250#63131250
    :param A:
    :return:
    """
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(C)
    eigval[eigval < 0.0] = 0.0
    res = eigvec @ np.diag(eigval) @ eigvec.T
    return res


@jit
def jax_project_S_plus(A):
    """

    :param A:
    :return:
    """
    eigval, eigvec = jnp.linalg.eigh(A)
    """
    if jnp.any(eigval < 0.0):
        print("projection")
    """
    eigval = jnp.where(eigval > 0.0, eigval, 0.0)
    res = eigvec @ jnp.diag(eigval) @ eigvec.T
    return res


def jax_project_S_shifted(A, r):
    """
    A + rI >=0

    :param A: symmetric matrix
    """
    assert r > 0.0, "r must be positive"
    assert A.shape[0] == A.shape[1], "A is not normal "
    C = A + r * jnp.identity(A.shape[0])
    eigval, eigvec = jnp.linalg.eigh(C)
    eigval = jnp.where(eigval > 0.0, eigval, 0.0)
    res = eigvec @ jnp.diag(eigval) @ eigvec.T
    # TODO: check this
    res -= r * jnp.identity(A.shape[0])
    return res


@jit
def jax_vec2_S_plus(vec):
    """

    :param ves:
    :return:
    """
    mat = jit_create_symmetric(vec)
    mat = jax_project_S_plus(mat)
    return sym2vec(mat)


def jax_vec2_S_shifted(vec, r):
    """

    :param vec:
    :param r:
    :return:
    """
    mat = jit_create_symmetric(vec)
    mat = jax_project_S_shifted(mat, r)
    return sym2vec(mat)


def jax_vec_projection(x, m, n, k):
    """

    :param m:
    :param n:
    :param k:
    :return:
    """
    size = k * (k + 1) // 2
    for i in range(m):
        x = jax.ops.index_update(x, jax.ops.index[size * i:size * (i + 1)], jax_vec2_S_plus(x[size * i:size * (i + 1)]))
    for j in range(n):
        x = jax.ops.index_update(x, jax.ops.index[size * (j + m):size * (j + m + 1)],
                                 jax_vec2_S_plus(x[size * (j + m):size * (j + m + 1)]))
    return x


def jax_vec_proj_shifted(x, m, n, k, r):
    """

    :param x:
    :param m:
    :param n:
    :param k:
    :param r: shift parameter
    :return:
    """
    size = k * (k + 1) // 2
    for i in range(m):
        x = jax.ops.index_update(x, jax.ops.index[size * i:size * (i + 1)],
                                 jax_vec2_S_shifted(x[size * i:size * (i + 1)], r))
    for j in range(n):
        x = jax.ops.index_update(x, jax.ops.index[size * (j + m):size * (j + m + 1)],
                                 jax_vec2_S_shifted(x[size * (j + m):size * (j + m + 1)], r))
    return x


def optimize(f, xinit, params, m, n, k, r=None, maxIter: int = 100, terminate_at_origin=False):
    """

    :param f:
    :param xinit:
    :param params:ｓ
    :param maxIter:
    :param verbose:
    :return:
    """
    grad_f = jit(jax.grad(f))
    Hess_f = jit(jax.hessian(f))
    f_vals = []
    x = xinit
    if r == None:
        projection_fun = jit(lambda val: jax_vec_projection(val, m, n, k))
    else:
        projection_fun = jit(lambda val: jax_vec_proj_shifted(val, m, n, k, r))
    for iter in range(maxIter):
        fval = f(x)
        f_vals.append(fval)
        x = x - 1.0 / params["L1"] * grad_f(x)
        x = projection_fun(x)
        print("#iterations: ", iter, " function value: ", fval,
              "Hess: ", jnp.linalg.norm(Hess_f(x)), "grad: ", jnp.linalg.norm(grad_f(x)))
        if terminate_at_origin and np.all(np.isclose(x, np.zeros_like(x), rtol=1e-30, atol=1e-30)):
            print("x is at origin")
            print(f(x))
            return
    return f_vals


def geninits(m, n, k, r=None):
    """ generate initial values

    :param m:
    :param n:
    :param k:
    :return:
    """
    # TODO: The treatment of r should be considered
    a_size = m * k * (k + 1) // 2
    b_size = n * k * (k + 1) // 2
    x = np.random.normal(0.0, 1.0, a_size + b_size)
    x = jax_vec_projection(x, m, n, k)
    return x


def is_nonconvex_at_origin(f, m, n, k, M):
    """

    :param f:
    :param m:
    :param n:
    :param k:
    :return:
    """
    size = (m + n) * k * (k + 1) // 2
    hess = jax.hessian(f)
    eigv = min_eigv(hess(np.zeros(size)))
    print("Problem size is:", size)
    print("Minimum eigen value at origin is:", eigv.item())
    print("Function value at origin is ", jnp.linalg.norm(M, ord='fro') ** 2)
    assert np.isclose(jnp.linalg.norm(M, ord='fro') ** 2, f(np.zeros(size)))
    return


def projected_gradient():
    # np_seed = int(input("Input seed "))
    np_seed = 5
    # np_seed = 40
    np.random.seed(np_seed)
    m = 5
    n = 5
    k = 4
    M, _, _ = gen_psf_params(m, n, k)
    # M = genM(M)
    frob = jit(lambda x: psf(m, n, k, M, x))
    xinit = geninits(m, n, k)
    # xinit = np.zeros((m + n) * k * (k + 1) // 2)
    params = dict()
    params["L1"] = 2000.0
    is_nonconvex_at_origin(frob, m, n, k, M)
    f_vals = optimize(frob, xinit, params, m, n, k, maxIter=500, terminate_at_origin=True)


def shifted_projected_gradient():
    # np_seed = int(input("Input seed "))
    np_seed = 3
    np.random.seed(np_seed)
    m = 5
    n = 5
    k = 4
    r = 0.3
    ratio = 0.2
    M, _, _ = gen_shifted_params(m, n, k, ratio, r)
    # M = genM(M)
    frob = jit(lambda x: psf(m, n, k, M, x))
    # xinit = geninits(m, n, k, r)
    xinit = np.zeros((m + n) * k * (k + 1) // 2)
    params = dict()
    params["L1"] = 1000.0
    is_nonconvex_at_origin(frob, m, n, k, M)
    f_vals = optimize(frob, xinit, params, m, n, k, r=r, maxIter=10000, terminate_at_origin=False)
    np.savetxt('outputs/random_init_pgd_hess.txt', f_vals, delimiter=',')  # X is an array


if __name__ == '__main__':
    shifted_projected_gradient()
