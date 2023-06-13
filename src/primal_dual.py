#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Primal dual interior point method
"""
import csv
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.linalg import LinAlgWarning

from src.utils import (
    pos_sym_inv,
    min_eigv,
    is_pd,
    jax_pos_sym_inv,
    Color,
    mask_to_Newton,
    sym2vec,
    create_symmetric,
    modified_bfgs_update,
)

jax.config.update("jax_enable_x64", True)

BACKTRACK = 0.8


def gen_merit(x, Z, f, X, nu, mu):
    """
    ψμ(x, Z) = F_BP(x) + nu * (F_PD(x, Z))
    :return:
    """

    # assert is_pd(Z), "Z is not positive definite"
    # assert is_pd(X(x)), "X(x) is not positive definite"
    def F_BP(x):
        return f(x) - mu * jnp.linalg.slogdet(X(x))[1]

    def F_PD(x, Z):
        return (
            jnp.trace(X(x) @ Z)
            - mu * jnp.linalg.slogdet(X(x))[1]
            - mu * jnp.linalg.slogdet(Z)[1]
        )

    return F_BP(x) + nu * F_PD(x, Z)


def is_approx_stationary(x, X, Z, grad_x, mu, params):
    ZF = np.linalg.norm(Z, ord="fro")
    invXF = np.linalg.norm(pos_sym_inv(X(x)), ord="fro")
    left = np.linalg.norm(grad_x)
    right = params["epsilon_g"] * (1 + mu * invXF + ZF)
    # print(left, right)
    return left <= right


def is_approx_comple(Z, grad_Z, mu, params):
    # invZF = jnp.linalg.norm(pos_sym_inv(Z), ord='fro')
    # assert is_pd(Z), "Z is not positive definite"
    invZF = np.linalg.norm(jax_pos_sym_inv(Z), ord="fro")
    left = np.linalg.norm(grad_Z, ord="fro")
    right = params["epsilon_mu"] * (1 + mu * invZF)
    # print("left:", left, "right:", right)
    return left <= right


def is_approx_sosp(x, X, Z, mineigval, mu, params):
    ZF = np.linalg.norm(Z, ord="fro")
    invXF = np.linalg.norm(pos_sym_inv(X(x)), ord="fro")
    right = -1 * params["epsilon_H"] * (1 + mu * invXF + ZF) ** 2
    print("Mineigval: ", mineigval, "right: ", right)
    return mineigval >= right


"""
def Newton_step(x, Z, X, psi, grad_psi_x, hess_x, params):
    B = mask_to_Newton(hess_x, a=1e-100, b=1e100)
    newton_direction = sp.linalg.solve(B, -grad_psi_x, assume_a="sym")
    step_size = 1
    bt_cnt = 0
    while not is_pd(X(x + step_size * newton_direction)) or psi(x + step_size * newton_direction, Z) \
            > psi(x, Z) - (step_size / 100) * np.linalg.norm(newton_direction) ** 2:
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print(Color.BLUE, "stepsize:", step_size, Color.END)
    print("backtrack:", bt_cnt)
    return x + step_size * newton_direction
"""


def Newton_step(x, Z, X, psi, grad_psi, B, params):
    B = mask_to_Newton(B, a=1e-10, b=1e7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            newton_direction = sp.linalg.solve(B, -grad_psi(x, Z), assume_a="sym")
        except LinAlgWarning:
            print(Color.RED + "Catch Warning" + Color.END)
            B = np.identity(B.shape[0])
            newton_direction = -grad_psi(x, Z)
    step_size = 1
    bt_cnt = 0
    while not is_pd(X(x + step_size * newton_direction)):
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    inner_backtrack = bt_cnt
    while (
        not is_pd(X(x + step_size * newton_direction))
        or psi(x + step_size * newton_direction, Z)
        > psi(x, Z) - (step_size / 100000) * np.linalg.norm(newton_direction) ** 2
    ):
        if bt_cnt - inner_backtrack > 10:
            break
        step_size = BACKTRACK * step_size
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    print(Color.BLUE, "stepsize:", step_size, Color.END)
    B = modified_bfgs_update(
        B,
        step_size * newton_direction,
        grad_psi(x + step_size * newton_direction, Z) - grad_psi(x, Z),
    )
    return x + step_size * newton_direction, B


def gradient_update_x(x, X, Z, dx, merit, params):
    """

    :param x:
    :param Z:
    :param mu:
    :param nu:
    :param params:
    :return:
    """
    bt_cnt = 0
    stepsize = min_eigv(X(x)) / (2.0 * params["L0"] * np.linalg.norm(dx))
    while merit(x + stepsize * dx, Z) > merit(x, Z) - stepsize / 2 * (
        np.linalg.norm(dx) ** 2
    ) or not is_pd(X(x + stepsize * dx)):
        stepsize = BACKTRACK * stepsize
        bt_cnt += 1
    print("backtrack:", bt_cnt)
    return x + stepsize * dx


def gradient_update_Z(x, Z, merit, dZ):
    # merit_fixed_x = merit_fixedx(x, merit)
    # dZ = -jax.grad(merit_fixed_x)(Z)
    stepsize = min_eigv(Z) / (2 * np.linalg.norm(dZ, ord="fro"))
    bt_cnt = 0
    while merit(x, Z + stepsize * dZ) > merit(x, Z) - stepsize / 2 * (
        np.linalg.norm(dZ, ord="fro") ** 2
    ):
        bt_cnt += 1
        stepsize = BACKTRACK * stepsize
    newZ = Z + stepsize * dZ
    # newZ = (newZ + newZ.T)/2
    print("backtrack: ", bt_cnt)
    return newZ


def negative_curvature(x, X, Z, merit, mineig_dir, mineig_val, params):
    assert mineig_val < 0, "eigen value is not negative"
    stepsize = min_eigv(X(x)) / (2 * params["L0"])
    while (
        merit(x + stepsize * mineig_dir, Z)
        > merit(x, Z) + stepsize**2 * (mineig_val) / 6
    ):
        stepsize = BACKTRACK * stepsize
    return x + stepsize * mineig_dir


def Lagrangian(x, Z, f, X):
    return f(x) - jnp.trace(X(x) @ Z)


def KKTresidual(x, grad_Lagrange, X, Z):
    """
    \sqrt{||∇_x L(x, Λ)||^2 + ||X(x)Λ||_F^2}

    """
    # Calculate Lambda
    assert is_pd(X(x))
    n = x.shape[0]
    m = Z.shape[0]
    return np.linalg.norm(grad_Lagrange(x, Z), ord=1) / n + np.trace(X(x) @ Z) / m


def inner_iteration(
    x,
    Z,
    f,
    X,
    nu,
    mu,
    params,
    fvals,
    nc,
    newton,
    _merit,
    _gradx,
    _gradZ,
    _Hessxx,
    _grad_Lagrange,
    csvwriter,
    start_time,
    maxIter=100000,
):
    """

    :param x:
    :param Z:
    :param f:lambda function
    :param X:lambda function
    :param params:
    :return:
    """

    def merit(_x, _Z):
        return _merit(_x, _Z, nu, mu)

    def gradx(_x, _Z):
        return _gradx(_x, _Z, nu, mu)

    def gradZ(_x, _Z):
        return _gradZ(_x, _Z, nu, mu)

    def Hessxx(_x, _Z):
        return _Hessxx(_x, _Z, nu, mu)

    curIter = 0
    # Z = mu * pos_sym_inv(X(x))
    last_step_is_dual = 0
    B = np.identity(x.shape[0])
    while curIter < maxIter:
        fvals.append(f(x))
        if is_approx_comple(Z, gradZ(x, Z), mu, params) == False:
            # Z direction
            step_start = time.time()
            curIter += 1
            Z = gradient_update_Z(x, Z, merit, -gradZ(x, Z))

            print(
                "Dual: Itr: ",
                curIter,
                " merit: ",
                merit(x, Z),
                "fval:",
                f(x),
                " KKT: residual:",
                str(KKTresidual(x, _grad_Lagrange, X, Z)),
            )
            now = time.time()
            csvwriter.writerow(["dual", f(x), now - start_time, now - step_start])
            last_step_is_dual += 1
        elif is_approx_stationary(x, X, Z, gradx(x, Z), mu, params) == False:
            # x direction
            curIter += 1
            step_start = time.time()
            if newton == True:
                x, B = Newton_step(x, Z, X, merit, gradx, B, params)
            else:
                x = gradient_update_x(x, X, Z, -gradx(x, Z), merit, params)
            print(
                Color.BLUE
                + "Primal: Itr: "
                + str(curIter)
                + " merit: "
                + str(merit(x, Z))
                + "fval:"
                + str(f(x))
                + " KKT: residual:"
                + str(KKTresidual(x, _grad_Lagrange, X, Z))
                + Color.END
            )
            now = time.time()
            csvwriter.writerow(["primal", f(x), now - start_time, now - step_start])
            last_step_is_dual = 0
        elif nc == False:
            print("innter iteration end")
            print("Itr: ", curIter, " merit: ", merit(x, Z), "fval:", f(x))
            return x, Z, curIter
        else:
            last_step_is_dual = 0
            step_start = time.time()
            hess = Hessxx(x, Z)
            lmin, dx = sp.linalg.eigh(hess, eigvals_only=False, subset_by_index=[0, 0])
            lmin = lmin.item()
            dx = dx[:, 0]
            if jnp.dot(dx, gradx(x, Z)) > 0:
                # dx must be a descent direction
                dx = -dx
            if is_approx_sosp(x, X, Z, lmin, mu, params) == False:
                x = negative_curvature(x, X, Z, merit, dx, lmin, params)
                curIter += 1
                now = time.time()
                csvwriter.writerow(["nc", f(x), now - start_time, now - step_start])
                print(
                    Color.GREEN
                    + "Negative curvature: Itr: "
                    + str(curIter)
                    + " merit: "
                    + str(merit(x, Z))
                    + " KKT: residual: "
                    + str(KKTresidual(x, _grad_Lagrange, X, Z))
                    + Color.END
                )
            else:
                print("innter iteration end")
                print("Itr: ", curIter, " merit: ", merit(x, Z), "fval:", f(x))
                return x, Z, curIter
    return x, Z, maxIter


def outer_iteration(
    f,
    X,
    x,
    Z,
    muinit,
    params,
    FILE_NAME="outputs/test.csv",
    muend=1e-30,
    nc=False,
    newton=False,
    max_iter=300,
):
    """

    :param f:
    :param X:
    :param x:
    :param Z:
    :param muinit:
    :param params:
    :param muend:
    :return:
    """
    mu = muinit
    print(Color.YELLOW + "nc:" + str(nc) + Color.END)
    print("muend:", muend)

    @jax.jit
    def merit(_x, _Z, _nu, _mu):
        return gen_merit(_x, _Z, f, X, _nu, _mu)

    @jax.jit
    def grad_Lagrange(_x, _Z):
        return jax.grad(Lagrangian, argnums=0)(_x, _Z, f, X)

    @jax.jit
    def gradx(_x, _Z, _nu, _mu):
        return jax.grad(merit, argnums=0)(_x, _Z, _nu, _mu)

    def gradZ(_x, _Z, _nu, _mu):
        return _nu * (X(_x) - _mu * pos_sym_inv(_Z))

    @jax.jit
    def Hessxx(_x, _Z, _nu, _mu):
        return jax.hessian(merit, argnums=0)(_x, _Z, _nu, _mu)

    # JIT COMPILING FOR TIME COMPARISON
    print(Color.GREEN + "JIT COMPILE START" + Color.END)
    assert np.all(np.isclose(Z, create_symmetric(sym2vec(Z), Z.shape[0])))
    jit_start = time.time()
    Hessxx(x, Z, 0.5, 0.5)
    merit(x, Z, 0.5, 0.5)
    gradZ(x, Z, 0.5, 0.5)
    gradx(x, Z, 0.5, 0.5)
    grad_Lagrange(x, Z)
    print(Color.GREEN + "JIT COMPILE END: " + str(time.time() - jit_start) + Color.END)
    with open(FILE_NAME, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        finit = f(x)
        start_time = time.time()
        csvwriter.writerow(["Initial", finit, 0])
        while mu > muend and max_iter > 0:
            fvals = []
            print(
                Color.RED
                + "mu is:"
                + str(mu)
                + " function values:"
                + str(f(x))
                + Color.END
            )
            assert is_pd(Z), "Z is not positive definite"
            assert is_pd(X(x)), "X(x) is not positive definite"
            mu = min(0.8 * mu, 10.0 * pow(mu, 1.5))
            nu = pow(mu, 0.1)
            params["epsilon_g"] = mu
            params["epsilon_H"] = mu
            params["epsilon_mu"] = pow(mu, 1.2)
            csvwriter.writerow(["# mu is " + str(mu)])
            x, Z, cnt = inner_iteration(
                x,
                Z,
                f,
                X,
                nu,
                mu,
                params,
                fvals,
                nc,
                newton,
                merit,
                gradx,
                gradZ,
                Hessxx,
                grad_Lagrange,
                csvwriter,
                start_time,
                max_iter,
            )
            max_iter -= cnt
    return
