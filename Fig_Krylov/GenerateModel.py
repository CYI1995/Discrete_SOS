import numpy  as np
import scipy
import math
import scipy.linalg as la
import matplotlib
from matplotlib import pyplot as plt 
import source as mycode 

def generate_all_majoranas(n_sites):
    """
    Generates all 2N Majorana operators for a system of n_sites.
    Returns: A list of 2N NumPy arrays.
    """
    majoranas = []
    
    # Fundamental 2x2 building blocks
    d = np.array([[0, 1], [0, 0]])
    z = np.array([[1, 0], [0, -1]])
    eye = np.eye(2)

    for j in range(n_sites):
        # Construct the annihilation operator c_j for the current site
        # c_j = Z^0 \otimes Z^1 \otimes ... \otimes d^j \otimes I^{j+1} ...
        c_j = np.array([[1]]) # Start with a scalar for the Kronecker product
        
        for i in range(n_sites):
            if i < j:
                c_j = np.kron(c_j, z)
            elif i == j:
                c_j = np.kron(c_j, d)
            else:
                c_j = np.kron(c_j, eye)
        
        # Site j provides two Majorana operators: 
        # m = 2j (gamma_a) and m = 2j + 1 (gamma_b)
        g_even = c_j + c_j.conj().T          # gamma_{2j}
        g_odd = -1j * (c_j - c_j.conj().T)   # gamma_{2j+1}
        
        majoranas.append(g_even)
        majoranas.append(g_odd)
        
    return majoranas

n = 6
n2 = int(2*n)
d = int(2**n)
d2 = int(2**n2)
majorana_ops = generate_all_majoranas(n)


mh = np.random.normal(0,1, size = (n2,n2))
mh = 1j * (mh - mh.T)
mh_norm = la.norm(mh,2)
mh = mh/mh_norm

FF_ham = np.zeros((d,d),dtype = complex)
for a in range(n2):
    wa = majorana_ops[a]
    for b in range(n2):
        wb = majorana_ops[b]

        FF_ham += 0.5 * mh[a][b] * wa @ wb

Id = np.eye(d, dtype=complex)

# Construct fermionic number operators n_j = c_j^\dagger c_j
number_ops = []

for j in range(n):
    gamma_even = majorana_ops[2 * j]
    gamma_odd = majorana_ops[2 * j + 1]

    c_j = 0.5 * (gamma_even + 1j * gamma_odd)
    n_j = c_j.conj().T @ c_j

    number_ops.append(n_j)

# Nearest-neighbor density-density interaction
interacting_ham = np.zeros((d,d),dtype = complex)
for j in range(n - 1):
    nj_centered = number_ops[j] - 0.5 * Id
    nk_centered = number_ops[j + 1] - 0.5 * Id

    interacting_ham += nj_centered @ nk_centered

# Remove tiny numerical non-Hermiticity
interacting_ham = 0.5 * (
    interacting_ham + interacting_ham.conj().T
)

np.save('FF_ham.npy',FF_ham)
np.save('Int_ham.npy',interacting_ham)