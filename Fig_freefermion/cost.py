import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import source as mycode


n = 100
n2 = 2 * n
mh = np.load("mh.npy")


def func_gaussian(x):
    return math.exp(-2 * x * x)


def func_id(x):
    return 1


def func_opt(x):
    return 1 / math.sqrt(math.cosh(2 * x))


def compare_gaps_vs_beta(beta_list, mh):

    eigs_full = la.eigvalsh(mh)

    if len(eigs_full) % 2 != 0:
        raise ValueError("The dimension of mh must be even.")

    N = len(eigs_full) // 2

    # Check that the spectrum has the expected ± pairing.
    if not np.allclose(
        eigs_full[N:],
        -eigs_full[:N][::-1],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "The spectrum of mh does not have the expected ± pairing."
        )

    # Keep one eigenvalue from each ± pair.
    # If spec(mh) = {±lambda_n/2}, then eigs contains lambda_n/2.
    eigs = eigs_full[N:]
    L = len(eigs)

    cost_SGauss = []
    cost_SI = []
    cost_Sopt = []

    for beta in beta_list:
        print("beta =", beta)

        List_for_SGauss = np.zeros(L)
        List_for_S0 = np.zeros(L)
        List_for_Sopt = np.zeros(L)

        for l in range(L):
            # eigs[l] = lambda_l / 2, so this equals beta * lambda_l.
            lambda_l = 2 * beta * eigs[l]

            List_for_SGauss[l] = (
                func_gaussian(0.5 * lambda_l) ** 2
                * math.cosh(lambda_l)
            )

            List_for_S0[l] = (
                func_id(0.5 * lambda_l) ** 2
                * math.cosh(lambda_l)
            )

            # This expression is analytically equal to 1.
            List_for_Sopt[l] = 1.0

        M_Gauss = 4 * np.sum(List_for_SGauss)
        gap_Gauss = 2 * np.min(List_for_SGauss)

        M_I = 4 * np.sum(List_for_S0)
        gap_I = 2 * np.min(List_for_S0)

        M_opt = 4 * np.sum(List_for_Sopt)
        gap_opt = 2 * np.min(List_for_Sopt)

        cost_SGauss.append(
            beta * math.sqrt(M_Gauss / gap_Gauss)
        )
        cost_SI.append(
            beta * math.sqrt(M_I / gap_I)
        )
        cost_Sopt.append(
            beta * math.sqrt(M_opt / gap_opt)
        )

    return {
        "beta": np.array(beta_list),
        "cost_SGauss": np.array(cost_SGauss),
        "cost_SI": np.array(cost_SI),
        "cost_Sopt": np.array(cost_Sopt),
    }



if mh.shape != (n2, n2):
    raise ValueError(
        f"Expected mh to have shape {(n2, n2)}, "
        f"but found {mh.shape}."
    )

beta_list = np.linspace(1, 200, 21)
data = compare_gaps_vs_beta(beta_list, mh)
np.save("data_cost.npy", data)