#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Negative Curvature interior point method

minimize f(x) x ∈ R^n,
subject to X(x) ≽ 0
"""
import csv
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.linalg import LinAlgWarning
from jax.config import config

from src.utils import create_symmetric, min_eigv, is_pd, pos_sym_inv, Color, mask_to_Newton, modified_bfgs_update

config.update("jax_enable_x64", True)

BACKTRACK = 0.8


def gen_psi(f, X, mu: float):
    return lambda x: f(x) - mu * jnp.linalg.slogdet(X(x))[1]


def fosp(grad_psi_x, Y, epsilon_g, verbose=False):
    """　check the first order stationary condition
    ∥∇ψμ(xk)∥2 <= εg(1 +trace(Y))

    Return True at FOSP
    :param psi:jax.grad(psi) at x
    :param Y:
    :param epsilon_g:
    :return:
    """
    left = sp.linalg.norm(grad_psi_x, ord=2)
    right = epsilon_g * (1.0 + np.trace(Y))
    # res = (left < right)
    res = left < epsilon_g
    if verbose:
        print("First order optimality left: ", left, "right:", right)
    return res


def sosp(lmin, Y, epsilon_H, verbose=False):
    """ check the second order stationary condition
    λmin(∇^2ψμ)≥−εH(1 +∥Y∥F)^2

    Return True at SOSP
    :param lmin: lambda min
    :param Y:
    :param epsilon_H:
    :return:
    """
    # right = - epsilon_H * (1.0 + np.linalg.norm(Y, ord='fro')) ** 2
    right = - epsilon_H * (1.0 + np.trace(Y)) ** 2
    if verbose:
        print("lmin: ", lmin[0], " right: ", right)
    return lmin > right


def gradient_step(x, X, psi, grad_psi_x, params):
    """ step using the gradient

    :return:
    """
    dx = - grad_psi_x
    step_size = min_eigv(X(x)) / (2.0 * params["L0"] * np.linalg.norm(dx))
    bt_cnt = 0
    while psi(x + step_size * dx) \
            > psi(x) - (step_size / 2) * np.linalg.norm(dx) ** 2:
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    return x + step_size * dx


def Newton_step(x, X, psi, grad_psi, B, params):
    B = mask_to_Newton(B, a=1e-10, b=1e7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            newton_direction = sp.linalg.solve(B, -grad_psi(x), assume_a="sym")
        except LinAlgWarning:
            print(Color.RED+"Catch Warning"+Color.END)
            B = np.identity(B.shape[0])
            newton_direction = -grad_psi(x)
    step_size = 1
    bt_cnt = 0
    while not is_pd(X(x + step_size * newton_direction)):
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    inner_backtrack = bt_cnt
    while not is_pd(X(x + step_size * newton_direction)) or psi(x + step_size * newton_direction) \
            > psi(x) - (step_size / 100000) * np.linalg.norm(newton_direction) ** 2:
        if bt_cnt -inner_backtrack >10:
            break
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    print(Color.BLUE, "stepsize:", step_size, Color.END)
    B = modified_bfgs_update(B, step_size * newton_direction, grad_psi(x+ step_size * newton_direction) - grad_psi(x))
    return x + step_size * newton_direction, B


def eigvec_step(x, dx, X, psi, grad_psi, params, lmin):
    if np.dot(dx, grad_psi(x)) > 0:
        dx = -dx
    step_size = min_eigv(X(x)) / (2 * params["L0"] * np.linalg.norm(dx))
    assert lmin < 0
    bt_cnt = 0
    while psi(x + step_size * dx) > psi(x) + step_size ** 2 / 6 * lmin:
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    return x + step_size * dx


def KKTresidual(x, grad_psi, X, Y):
    """
    \sqrt{||∇_x L(x, Λ)||^2 + ||X(x)Λ||_F^2}

    """
    n = x.shape[0]
    m = Y.shape[0]
    return np.linalg.norm(grad_psi(x), ord=1)/n + np.trace(X(x) @ Y)/m


def inner_iteration(f, X, x, mu, params, _merit, _grad_psi, _Hess_psi, csvwriter, start_time, nc, newton, maxIter: int = 100, verbose=False):
    def psi(_x):
        return _merit(_x, mu)

    def grad_psi(_x):
        return _grad_psi(_x, mu)

    def Hess_psi(_x):
        return _Hess_psi(_x, mu)

    """
    def test_hessian(x):
        dx = np.random.rand(*x.shape)
        dx *= 1e-7
        hess = Hess_psi(x)
        rand_hess = np.random.rand(x.shape[0], x.shape[0])
        rand_hess = np.linalg.norm(rand_hess)/np.linalg.norm(hess)
        assert np.all(np.isclose(hess @ dx,
                                 grad_psi(x + dx) - grad_psi(x))), str(hess @ dx-(grad_psi(x + dx) - grad_psi(x))/rand_hess*np.random.rand(x.shape[0], x.shape[0]) @ dx-(grad_psi(x + dx) - grad_psi(x)))
    test_hessian(x)
    """
    B = np.identity(x.shape[0])
    for cnt in range(maxIter):
        # assert min_eigv(X(x)) * np.trace(pos_sym_inv(X(x))) >= 1, str(min_eigv(X(x)) * np.trace(pos_sym_inv(X(x))))
        # assert min_eigv(X(x)) * np.trace(Y) >= mu, "λmin(X) * trace(Y) >= μ"
        grad_psi_x = grad_psi(x)
        Y = mu * pos_sym_inv(X(x))
        if fosp(grad_psi_x, Y, params["epsilon_g"], verbose) == False:
            if newton == True:
                x, B = Newton_step(x, X, psi, grad_psi, B, params)
            else:
                x = gradient_step(x, X, psi, grad_psi_x, params)
            csvwriter.writerow(["gradient", f(x), time.time() - start_time])
        else:
            print(Color.YELLOW+"First order stationary point"+Color.END)
            Hess_psi_x = Hess_psi(x)
            print("Start eigen decomp size=", str(Hess_psi_x.shape))
            lmin, dx = sp.linalg.eigh(
                Hess_psi_x, eigvals_only=False, subset_by_index=[0, 0])
            print("end eigen decomp")
            dx = dx[:, 0]
            if nc and not sosp(lmin, Y, params["epsilon_H"], verbose):
                print(Color.GREEN + "Negative eigenvalue direction" + Color.END)
                x = eigvec_step(x, dx, X, psi, grad_psi, params, lmin)
                csvwriter.writerow(["nc", f(x), time.time() - start_time])
            else:
                # terminate
                print(Color.RED + "second order stationary point" + Color.END)
                return x, cnt
        assert is_pd(X(x)), "X(x) is not positive definite"
        print("mu, iter: ", mu, cnt-1, "Ψ val:", psi(x), " KKT residualL ", KKTresidual(x, grad_psi, X, Y), " f val: ",
              f(x), "gradx", np.linalg.norm(grad_psi(x)))
    return x, maxIter


def outer_iteration(f, X, x, muinit, params, FILE_NAME="outputs/test.csv", muend=1e-3, nc=True, newton=False, max_iter=1000):
    mu = muinit

    @jax.jit
    def merit(_x, _mu):
        return gen_psi(f, X, _mu)(_x)

    @jax.jit
    def grad_psi(_x, _mu):
        return jax.grad(merit, argnums=0)(_x, _mu)

    @jax.jit
    def Hess_psi(_x, _mu):
        return jax.hessian(merit, argnums=0)(_x, _mu)

    # JIT COMPILING FOR TIME COMPARISON
    assert is_pd(X(x)), "X(x) is not positive definite"
    print(Color.GREEN + "JIT COMPILE START" + Color.END)
    print(Color.YELLOW + "nc:" + str(nc)+Color.END)
    print(Color.YELLOW + "Newton:" + str(newton)+Color.END)
    jit_start = time.time()
    f(x)
    merit(x, 0.5)
    grad_psi(x, 0.5)
    Hess_psi(x, 0.5)
    pos_sym_inv(X(x))
    print(Color.GREEN + "JIT COMPILE END: " +
          str(time.time() - jit_start) + Color.END)
    with open(FILE_NAME, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        finit = f(x)
        start_time = time.time()
        csvwriter.writerow(["Initial", finit, 0])
        while mu > muend and max_iter > 0:
            print(Color.RED + "mu is:" + str(mu) +
                  " function values:" + str(f(x)) + Color.END)
            assert is_pd(X(x))
            mu = min(0.8 * mu, 10.0 * pow(mu, 1.4))
            params["epsilon_g"] = mu
            # 0.1 * mu
            params["epsilon_H"] = mu
            csvwriter.writerow(["# mu is " + str(mu)])
            x, cnt = inner_iteration(f, X, x, mu, params, merit, grad_psi,
                                     Hess_psi, csvwriter, start_time, nc, newton, maxIter=max_iter, verbose=False)
            max_iter -= cnt
    return x


def test_solve():
    def f(x): return (x[0] - 2.0) * (x[0] - 2.) - (x[1] - 0.5) * (x[1] - 0.5) + \
        (x[2] - 1.) * (x[2] - 1.) + (x[3] - 4.) * (x[3] - 4.) + \
        (x[4] - 2.) * (x[4] - 2.) + (x[5] - 5.) * (x[5] - 5.)

    def X(x): return create_symmetric(x, 3)
    mu = 1.0
    xinit = np.array([1., 0.5, 0, 1., 0, 1])
    params = {}
    params["L0"] = 1
    outer_iteration(f, X, xinit, mu, params)


if __name__ == '__main__':
    test_solve()
