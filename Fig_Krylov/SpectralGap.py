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

def op_inner_product(X, Y):
    return np.vdot(X, Y)

def op_norm(X):
    return np.linalg.norm(X)

def Krylov_estimation(m, ham, x, opt, tol=1e-7):
    """
    Lanczos-Krylov subspace estimation for operator exponential: exp(x * [ham, ·]) (opt)
    Standard implementation for quantum operator Lie algebra propagation
    
    Parameters:
        m (int): Max Krylov subspace dimension
        ham (np.ndarray): Hamiltonian (square matrix)
        opt (np.ndarray): Target operator (same size as ham)
        tol (float): Convergence tolerance for Lanczos steps
    
    Returns:
        np.ndarray: Krylov-approximated exponentiated operator
    """
    # Get dimension (fixes undefined 'd')
    d = ham.shape[0]
    
    # --------------------------
    # Step 1: Initial Lanczos step
    # --------------------------
    # Commutator: [ham, opt] = ham @ opt - opt @ ham
    norm_opt = op_norm(opt)
    opt = opt/norm_opt
    A1 = ham @ opt - opt @ ham
    a1 = op_inner_product(opt, A1).real
    B1 = A1 - a1 * opt

    b1 = op_norm(B1)

    # Initialize lists
    a_list = [a1]
    b_list = [b1]
    Obs_list = [opt]  # Krylov basis vectors

    # Normalize first basis element
    if b1 > tol:
        B1 = B1 / b1
    else:
        B1 = np.zeros_like(opt)

    Obs_list.append(B1)
    B_pre = opt.copy()
    B_curr = B1.copy()
    old_b = b1

    # --------------------------
    # Step 2: Lanczos iteration (FIXED loop range)
    # --------------------------
    # Iterate to build m-dimensional subspace (range(m-1) fixes overcounting)
    for t in range(m - 1):
        # Commutator with HAM (fixed H0 -> ham mismatch)
        A_temp = ham @ B_curr - B_curr @ ham
        a_temp = op_inner_product(B_curr, A_temp).real
        
        # 3-term Lanczos recurrence
        B_next = A_temp - a_temp * B_curr - old_b * B_pre
        next_b = op_norm(B_next)
        
        # Store coefficients
        a_list.append(a_temp)
        b_list.append(next_b)
        
        # Normalize & advance basis
        if next_b > tol:
            B_next = B_next / next_b
            Obs_list.append(B_next)
            B_pre = B_curr.copy()
            B_curr = B_next.copy()
            old_b = next_b
        else:
            print(f"Krylov subspace closed at step {t}")
            break

    # --------------------------
    # Step 3: Build tridiagonal Lanczos matrix
    # --------------------------
    L = len(a_list)
    MatLanczos = np.zeros((L, L), dtype=complex)
    for l in range(L - 1):
        MatLanczos[l, l] = a_list[l]
        MatLanczos[l, l + 1] = b_list[l]
        MatLanczos[l + 1, l] = b_list[l]
    MatLanczos[L - 1, L - 1] = a_list[L - 1]

    # --------------------------
    # Step 4: Compute Krylov approximation
    # --------------------------
    v0 = np.zeros(L, dtype=complex)
    v0[0] = 1.0
    
    # Exponentiate small Lanczos matrix
    vecx = la.expm(x * MatLanczos) @ v0
    
    # Reconstruct operator from Krylov basis
    Obs_Lanczos = np.zeros((d, d), dtype=complex)
    for l in range(L):
        Obs_Lanczos += vecx[l] * Obs_list[l]

    return norm_opt * Obs_Lanczos


# ========== 全局固定参数 ==========
n = 4
d = int(2**n)
n2 = int(n*2)
d2 = int(4**n)
Id = np.eye(d)
majorana_ops = generate_all_majoranas(n)

# 加载哈密顿量（只加载一次，不用重复读文件）
H0  = np.load('FF_ham.npy')
V = np.load('Int_ham.npy')

m_list = np.array([12,24,36])

# 构造J算子集合（只构造一次）
SetofJ = []
for a in range(n):
    Xa = mycode.SingleX(a,n)
    Za = mycode.SingleZ(a,n)
    SetofJ.append(Xa)
    SetofJ.append(Za)

L = 20
Listofbeta = np.linspace(1/L,8,L) 

# 四个不同h，循环分别计算并保存
h_list = np.array([0.01, 0.1, 1, 10])

# 外层循环遍历4个h
for h in h_list:
    print(f"===== 当前计算 h = {h} =====")
    # 每个h独立初始化保真度数组，互不干扰
    Listofspectralgap = np.zeros((L,4))
    
    # 构造当前h对应的总哈密顿量
    FH_ham = H0 + h * V

    # 遍历所有beta
    for l in range(L):
        print(f"beta 循环 l = {l}")
        beta = Listofbeta[l]
        MH = np.zeros((d2,d2),dtype = complex)
        Id = np.eye(d)
        
        # 精确MH
        for a in range(n2):
            Ja = SetofJ[a]
            La = la.expm(-0.25*beta*FH_ham) @ Ja @ la.expm(0.25*beta*FH_ham)
            Ra = la.expm(0.25*beta*FH_ham) @ Ja @ la.expm(-0.25*beta*FH_ham)
            Ga = np.kron(La,Id) - np.kron(Id,Ra.T)
            MH += Ga.conj().T @ Ga 

        eig = la.eigvalsh(0.5*MH)
        sg = eig[1] - eig[0]
        Listofspectralgap[l][0] = sg 

        # 三个不同m的Krylov近似
        for i in range(3):
            m = int(m_list[i])
            MH_Krylov = np.zeros((d2,d2),dtype = complex)   
            for a in range(n2):
                Ja = SetofJ[a]
                La = Krylov_estimation(m, FH_ham, -0.25*beta, Ja)
                Ra = Krylov_estimation(m, FH_ham, 0.25*beta, Ja)
                KGa = np.kron(La,Id) - np.kron(Id,Ra.T)
                MH_Krylov += KGa.conj().T @ KGa 

            eigk = la.eigvalsh(0.5*MH_Krylov)
            sg_Krylov = eigk[1] - eigk[0]
            Listofspectralgap[l][i+1] = sg_Krylov 

    # 按h值命名保存npy，4个文件互不覆盖
    save_name = f"Listofspectralgap_h{h}.npy"
    np.save(save_name, Listofspectralgap)
    print(f"已保存 h={h} 数据至: {save_name}\n")