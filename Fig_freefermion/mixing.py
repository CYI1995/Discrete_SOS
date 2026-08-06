import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import source as mycode

beta = 100.0
n = 100
n2 = 2 * n

mh = np.load("mh.npy")


def func_gaussian(x):
    return math.exp(-2 * x * x)


def func_id(x):
    return 1


def func_opt(x):
    return 1 / math.sqrt(math.cosh(2 * x))


def compare_gaps_vs_beta(time_list, mh):

    eigs_full = la.eigvalsh(mh)

    if len(eigs_full) % 2 != 0:
        raise ValueError("The dimension of mh must be even.")

    L = len(eigs_full) // 2

    if not np.allclose(
        eigs_full[L:],
        -eigs_full[:L][::-1],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "The spectrum of mh does not have the expected ± pairing."
        )

    # Keep one representative from each ± pair.
    # Each eigs[l] equals lambda_l / 2.
    eigs = eigs_full[L:]

    List_for_SGauss = np.zeros(L)
    List_for_S0 = np.zeros(L)
    List_for_Sopt = np.zeros(L)

    for l in range(L):
        lambda_l = 2 * beta * eigs[l]

        List_for_SGauss[l] = (
            func_gaussian(0.5 * lambda_l) ** 2
            * math.cosh(lambda_l)
        )

        List_for_S0[l] = (
            func_id(0.5 * lambda_l) ** 2
            * math.cosh(lambda_l)
        )

        # Exactly equal to one analytically.
        List_for_Sopt[l] = 1.0

    M_Gauss = 4 * np.sum(List_for_SGauss)
    M_I = 4 * np.sum(List_for_S0)
    M_opt = 4 * np.sum(List_for_Sopt)

    mixing_SGauss = []
    mixing_SI = []
    mixing_Sopt = []

    for t in time_list:
        print("t =", t)

        List_for_SGauss = np.zeros(L)
        List_for_S0 = np.zeros(L)
        List_for_Sopt = np.zeros(L)

        for l in range(L):
            lambda_l = 2 * beta * eigs[l]

            weight_gauss = (
                func_gaussian(0.5 * lambda_l) ** 2
                * math.cosh(lambda_l)
            )
            Gamma_Gauss = weight_gauss / M_Gauss
            List_for_SGauss[l] = (
                abs(math.tanh(lambda_l))
                * math.exp(-4 * Gamma_Gauss * t)
            )

            weight_id = math.cosh(lambda_l)
            Gamma_id = weight_id / M_I
            List_for_S0[l] = (
                abs(math.tanh(lambda_l))
                * math.exp(-4 * Gamma_id * t)
            )

            weight_opt = 1.0
            Gamma_opt = weight_opt / M_opt
            List_for_Sopt[l] = (
                abs(math.tanh(lambda_l))
                * math.exp(-4 * Gamma_opt * t)
            )

        mixing_SGauss.append(
            0.5 * np.sum(List_for_SGauss)
        )
        mixing_SI.append(
            0.5 * np.sum(List_for_S0)
        )
        mixing_Sopt.append(
            0.5 * np.sum(List_for_Sopt)
        )

    return {
        "time": np.array(time_list),
        "mixing_SGauss": np.array(mixing_SGauss),
        "mixing_SI": np.array(mixing_SI),
        "mixing_Sopt": np.array(mixing_Sopt),
    }


if mh.shape != (n2, n2):
    raise ValueError(
        f"Expected mh to have shape {(n2, n2)}, "
        f"but found {mh.shape}."
    )

time_list = np.linspace(1, 1000, 21)
data = compare_gaps_vs_beta(time_list, mh)

np.save("data_mixing.npy", data)