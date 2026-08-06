import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import source as mycode


def func_gaussian(x):
    return math.exp(-2*x*x)

def func_id(x):
    return 1

def func_opt(x):
    return 1/math.sqrt(math.cosh(2*x))

def compare_gaps_vs_beta(beta_list, mh):

    eigs = la.eigvalsh(mh)
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
            lambda_l = 2 * beta * eigs[l]
            List_for_SGauss[l] = func_gaussian(0.5 * lambda_l)**2 * math.cosh(lambda_l)
            List_for_S0[l] = func_id(0.5 * lambda_l)**2 * math.cosh(lambda_l)
            List_for_Sopt[l] = func_opt(0.5 * lambda_l)**2 * math.cosh(lambda_l)

        M_Gauss = 4 * np.sum(List_for_SGauss)
        gap_Gauss = 2 * np.min(List_for_SGauss)
        M_I = 4 * np.sum(List_for_S0)
        gap_I = 2 * np.min(List_for_S0)
        M_opt = 4 * np.sum(List_for_Sopt)
        gap_opt = 2 * np.min(List_for_Sopt)

        cost_SGauss.append(beta * math.sqrt(M_Gauss/(2*gap_Gauss)))
        cost_SI.append(beta * math.sqrt(M_I/(2*gap_I)))
        cost_Sopt.append(beta * math.sqrt(M_opt/(2*gap_opt)))

    return {
        "beta": np.array(beta_list),
        "cost_SGauss": np.array(cost_SGauss),
        "cost_SI": np.array(cost_SI),
        "cost_Sopt": np.array(cost_Sopt),
    }


n = 100
n2 = 2*n

# mh = np.random.normal(0,1,size = (n2, n2))
# mh = 0.5j * (mh - mh.T) 
# eig = np.linalg.eigvalsh(mh)
# one_norm = 0
# for i in range(n2):
#     one_norm += abs(eig[i])
# mh = mh/one_norm
# mv = np.random.normal(0,1,size = (n,n))
# np.save('mh.npy',mh)

mh = np.load("mh.npy")
beta_list = np.linspace(1, 200, 21)
data = compare_gaps_vs_beta(beta_list, mh)
np.save('data_cost.npy',data)