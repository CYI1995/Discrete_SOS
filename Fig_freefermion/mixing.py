import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import source as mycode

beta = 100.0

def func_gaussian(x):
    return math.exp(-2*x*x)

def func_id(x):
    return 1

def func_opt(x):
    return 1/math.sqrt(math.cosh(2*x))

def compare_gaps_vs_beta(time_list, mh):

    eigs = la.eigvalsh(mh)
    L =  int(len(eigs)/2)

    List_for_SGauss = np.zeros(L)
    List_for_S0 = np.zeros(L)
    List_for_Sopt = np.zeros(L)

    for l in range(L):
        lambda_l = 2 * beta * eigs[l]
        List_for_SGauss[l] = func_gaussian(0.5 * lambda_l)**2 * math.cosh(lambda_l)
        List_for_S0[l] = func_id(0.5 * lambda_l)**2 * math.cosh(lambda_l)
        List_for_Sopt[l] = func_opt(0.5 * lambda_l)**2 * math.cosh(lambda_l)

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
            Gamma_Gauss = func_gaussian(0.5 * lambda_l)**2 * math.cosh(lambda_l) / M_Gauss
            List_for_SGauss[l] = abs(math.tanh(lambda_l)) * math.exp(-4*Gamma_Gauss*t)

            Gamma_id = func_id(0.5 * lambda_l)**2 * math.cosh(lambda_l) / M_I
            List_for_S0[l] = abs(math.tanh(lambda_l)) * math.exp(-4*Gamma_id*t)

            Gamma_opt = func_opt(0.5 * lambda_l)**2 * math.cosh(lambda_l) / M_opt
            List_for_Sopt[l] = abs(math.tanh(lambda_l)) * math.exp(-4*Gamma_opt*t)

        mixing_SGauss.append(0.5 * np.sum(List_for_SGauss))
        mixing_SI.append(0.5 * np.sum(List_for_S0))
        mixing_Sopt.append(0.5 * np.sum(List_for_Sopt))

    return {
        "time": np.array(time_list),
        "mixing_SGauss": np.array(mixing_SGauss),
        "mixing_SI": np.array(mixing_SI),
        "mixing_Sopt": np.array(mixing_Sopt),
    }


n = 100
n2 = 2*n

mh = np.load("mh.npy")
time_list = np.linspace(1, 1000, 21)
data = compare_gaps_vs_beta(time_list, mh)
np.save('data_mixing.npy',data)