#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Toy problems for Positive semidefinite matrix factorization

 minimize sum_i sum_j (X_{ij} - trace(AiBj))^2
 subject to Ai Bj are positive semidefinite matrix

 X is m times n

 Algorithms for positive semidefinite factorization
 https://link.springer.com/article/10.1007%2Fs10589-018-9998-x
"""
import argparse
import os
import time

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.config import config
from jax.scipy.linalg import block_diag

import src.primal as primal
import src.primal_dual as primal_dual
from src.utils import (
    gen_random_pd,
    sym2vec,
    jit_create_symmetric,
    mask_to_mat_lower,
    pos_sym_inv,
    Color,
    is_pd,
)

config.update("jax_enable_x64", True)


def gen_psf_params(m, n, k):
    """Positive semidefintie matrix factorization
    minimize sum_i sum_j (X_{ij} - trace(AiBj))^2
    subject to Ai Bj are positive semidefinite matrix

    :param m: i in [0, m-1]
    :param n: j in [0, n-1]
    :param k: matrix size
    :return:
    """
    As = []
    for i in range(m):
        As.append(gen_random_pd(k))
    Bs = []
    for j in range(n):
        Bs.append(gen_random_pd(k))
    X = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            X[i, j] = jnp.trace(As[i] @ Bs[j])
    return (X, As, Bs)


def gen_shifted_params(m, n, k, ratio, r):
    flag = True
    while flag:
        As = []
        for i in range(m):
            As.append(mask_to_mat_lower(gen_random_pd(k), ratio, r))
        Bs = []
        for j in range(n):
            Bs.append(mask_to_mat_lower(gen_random_pd(k), ratio, r))
        X = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                X[i, j] = np.trace(As[i] @ Bs[j])
        if np.all(X >= 0):
            flag = False
        else:
            flag = True
            print(Color.RED + "X is not nonnegative matrix" + Color.END)
            print(Color.RED + "Generate X again" + Color.END)
        return (X, As, Bs)


def psf(m, n, k: int, M, x):
    """objective function of psf
    :param m:
    :param n:
    :param k:
    :return: function
    """
    size = k * (k + 1) // 2
    assert x.shape[0] == (m + n) * size, "x shape mismatch"
    index = 0
    As = []
    for i in range(m):
        chosen_x = x[index : index + size]
        A = jit_create_symmetric(chosen_x)
        # A = create_symmetric(chosen_x, k)
        As.append(A)
        index += size
    Bs = []
    for j in range(n):
        # B = create_symmetric(x[index: index + size], k)
        B = jit_create_symmetric(x[index : index + size])
        Bs.append(B)
        index += size
    res = 0.0
    for i in range(m):
        for j in range(n):
            tr = jnp.trace(As[i] @ Bs[j])
            # print("trace: ", tr)
            to_squared = M[i, j] - tr
            # print("to_squred: ", to_squared)
            res += jnp.power(to_squared, 2)
    return res


def pscon(m, n, k, x):
    """constraints of the positive semidefinite matrix

    :param m:
    :param n:
    :param k:
    :return:
    """
    size = k * (k + 1) // 2
    assert x.shape[0] == (m + n) * size, "x shape mismatch"
    # make large single constraints
    mats = []
    index = 0
    for i in range(m):
        A = jit_create_symmetric(x[index : index + size])
        # A = create_symmetric(x[index:index + size], k)
        mats.append(A)
        index += size
    for j in range(n):
        B = jit_create_symmetric(x[index : index + size])
        # B = create_symmetric(x[index:index + size], k)
        mats.append(B)
        index += size
    return block_diag(*mats)


def shifted_pscon(m, n, k, x, r):
    """shifted constraitns

    A + rI>=0
    B + rI>=0

    :param m:
    :param n:
    :param k:
    :param x:
    :param r:  shift parameter
    :return:
    """
    size = k * (k + 1) // 2
    assert x.shape[0] == (m + n) * size, "x shape mismatch"
    # make large single constraints
    mats = []
    index = 0
    for i in range(m):
        A = jit_create_symmetric(x[index : index + size])
        # A = create_symmetric(x[index:index + size], k)
        mats.append(A)
        index += size
    for j in range(n):
        B = jit_create_symmetric(x[index : index + size])
        # B = create_symmetric(x[index:index + size], k)
        mats.append(B)
        index += size
    res = block_diag(*mats)
    res = res + r * jnp.identity(res.shape[0])
    return res


def dist2sol_fromAB(As, Bs, m, n, k, x):
    size = k * (k + 1) // 2
    assert x.shape[0] == (m + n) * size, "x shape mismatch"
    index = 0
    res = 0.0
    for i in range(m):
        res += jnp.linalg.norm(
            As[i] - jit_create_symmetric(x[index : index + size]), ord="fro"
        )
        index += size
    for j in range(n):
        res += jnp.linalg.norm(
            Bs[j] - jit_create_symmetric(x[index : index + size]), ord="fro"
        )
        index += size
    return res


@jit
def conv2x(As, Bs):
    """As and Bs to vector

    :param As:
    :param Bs:
    :return:
    """

    x = [None for i in range(len(As) + len(Bs))]
    for idx, A in enumerate(As):
        x[idx] = sym2vec(A)
    for idx, B in enumerate(Bs):
        x[idx + len(As)] = sym2vec(B)
    m = len(As)
    n = len(Bs)
    k = A.shape[0]
    res = jnp.concatenate(x)
    size = k * (k + 1) // 2 * (m + n)
    assert res.shape[0] == size, str(res.shape)
    return res


def make_init(m, n, k):
    # make initial point
    As = []
    for i in range(m):
        As.append(gen_random_pd(k))
    Bs = []
    for j in range(n):
        Bs.append(gen_random_pd(k))
    res = conv2x(As, Bs)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--muinit", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--nc", action="store_true")
    parser.add_argument("--Newton", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--muend", type=float, default=1e-3)
    parser.add_argument("--r", type=float, default=0.3)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    print(args)
    m = args.m
    n = args.n
    k = args.k
    r = args.r
    ratio = 0.2
    np.random.seed(args.seed)
    M, As, Bs = gen_shifted_params(m, n, k, ratio, r)
    f = jit(lambda x: psf(m, n, k, M, x))
    X = jit(lambda x: shifted_pscon(m, n, k, x, r))
    size = (m + n) * k * (k + 1) // 2
    print("problem_size:", size)
    xinit = np.zeros(size) + args.noise * np.random.rand(size)
    print("M shape: ", M.shape)
    np.savetxt("vinit.csv", M, delimiter=",")
    # xinit = make_init(m, n, k)
    muinit = args.muinit
    assert is_pd(X(xinit)), "NEEDS feasible initial point"
    params = dict()
    params["L0"] = 1.0
    print("finit:", f(xinit))
    print("muinit:", muinit)
    timestr = time.strftime("%m%d")
    if args.dual == True:
        Zinit = muinit * pos_sym_inv(X(xinit))
        print(Color.RED + "Primal dual Algorithm" + Color.END)
        fname = "outputs/psf/" + timestr + "/primal_dual/" + str(vars(args)) + ".csv"
        os.makedirs("outputs/psf/" + timestr + "/primal_dual/", exist_ok=True)
        primal_dual.outer_iteration(
            f,
            X,
            xinit,
            Zinit,
            muinit,
            params,
            FILE_NAME=fname,
            muend=args.muend,
            nc=args.nc,
            newton=args.Newton,
            max_iter=args.maxiter,
        )
    else:
        print(Color.RED + "Primal Algorithm" + Color.END)
        timestr = time.strftime("%m%d")
        fname = "outputs/psf/" + timestr + "/primal/" + str(vars(args)) + ".csv"
        os.makedirs("outputs/psf/" + timestr + "/primal/", exist_ok=True)
        primal.outer_iteration(
            f,
            X,
            xinit,
            muinit,
            params,
            FILE_NAME=fname,
            nc=args.nc,
            newton=args.Newton,
            muend=args.muend,
            max_iter=args.maxiter,
        )
