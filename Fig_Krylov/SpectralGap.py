"""
Spectral gap of the operator

    MH = sum_a Ga^dag Ga,
    Ga = La (x) I - I (x) Ra^T,
    La = e^{-beta H/4} J_a e^{+beta H/4} = exp(-x ad_H)(J_a),
    Ra = e^{+beta H/4} J_a e^{-beta H/4} = exp(+x ad_H)(J_a),   x = beta/4,

comparing the exact result against the Krylov approximation of La and Ra.
Reported quantity is  lambda_1 - lambda_0  of  MH / 2,  as before.

Structural changes relative to the direct implementation:

1.  The Krylov space K_m(ad_H, J_a) depends only on (H, J_a) -- not on x, and
    not on its sign.  One Lanczos run per operator therefore serves La, Ra,
    every beta and every m: 12 runs per h instead of 2 * 12 * 3 * 20.

2.  Ga^dag Ga is expanded before the Kronecker products are formed,

        Ga^dag Ga = (La^dag La) (x) I  +  I (x) (Ra^T)^dag(Ra^T)
                    - La^dag (x) Ra^T  -  (La^dag (x) Ra^T)^dag

    so every matrix product stays d x d.  The original code multiplied two
    d^2 x d^2 matrices for each (a, beta, m), which dominated the runtime.

3.  The factor exp(-x * W), W = lambda_max - lambda_min of H, is pulled out
    of both La and Ra.  MH is then computed in units of exp(-2 x W); this is
    the only way to reach large beta at all, since e^{beta H / 4} overflows
    float64 once beta * ||H|| / 4 > 709 and MH overflows at half that.

Because the gap is NOT scale invariant, the exponential factor is restored
at the end.  Two files are written per h:

    Listofspectralgap_h{h}.npy   (L, 4)  gap in the original units,
                                         = gap_scaled * exp(2 * x * W);
                                         may overflow to inf at large beta
    Listoflog10gap_h{h}.npy      (L, 4)  log10 of the same quantity, always
                                         finite -- use this for plotting

Column 0 is exact, columns 1..3 correspond to m_list.
"""

import numpy as np
import scipy.linalg as la

import source as mycode


# ----------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------

def matrix_norm(M):
    """Spectral norm (largest singular value).

    Computed by SVD rather than via eigvalsh(M^dag M): forming M^dag M
    squares the dynamic range and overflows to inf/nan whenever
    ||M|| exceeds ~1e154.
    """
    return la.norm(M, 2)


def hs_inner_product(X, Y):
    """Hilbert-Schmidt inner product Tr(X^dag Y)."""
    return np.vdot(X, Y)


def hs_norm(X):
    """Hilbert-Schmidt / Frobenius norm."""
    return la.norm(X, ord="fro")


# ----------------------------------------------------------------------
# Lanczos on operator space  (independent of x)
# ----------------------------------------------------------------------

def lanczos_tridiag(m, ham, opt, tol=1e-12, reorthogonalize=True):
    r"""
    Lanczos iteration for the self-adjoint superoperator ad_H acting on
    operator space with the Hilbert-Schmidt inner product, started from
    opt / ||opt||_HS.

    Returns
    -------
    basis : ndarray (k, d, d)
        Orthonormal Krylov basis, k <= m.
    alphas : ndarray (k,)
        Diagonal of the projected tridiagonal matrix T.
    betas : ndarray (k - 1,)
        Off-diagonal of T.
    norm_opt : float
        ||opt||_HS, the factor removed from the starting vector.

    The output is nested: the leading k x k block of T equals the T of a
    k-step run, so a single m = max(m_list) run covers all smaller m.
    """
    ham = np.asarray(ham, dtype=complex)
    opt = np.asarray(opt, dtype=complex)

    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    if ham.ndim != 2 or ham.shape[0] != ham.shape[1]:
        raise ValueError("ham must be a square matrix.")
    if opt.shape != ham.shape:
        raise ValueError("opt and ham must have the same shape.")
    if not np.allclose(ham, ham.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "ham must be Hermitian: Lanczos requires ad_H to be self-adjoint."
        )

    norm_opt = hs_norm(opt)
    if norm_opt == 0.0:
        return (
            np.zeros((0,) + opt.shape, dtype=complex),
            np.zeros(0),
            np.zeros(0),
            0.0,
        )

    # ||ad_H||_HS <= 2 ||H||_2 sets the natural scale for all coefficients.
    lanczos_scale = max(1.0, 2.0 * la.norm(ham, 2))

    q = opt / norm_opt
    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    basis = []
    alphas = []
    betas = []

    for j in range(m):
        basis.append(q.copy())

        z = ham @ q - q @ ham          # z = ad_H(q_j)
        if j > 0:
            z -= beta_prev * q_prev

        alpha = hs_inner_product(q, z)

        # ad_H self-adjoint  =>  alpha real.  The tolerance is set by the
        # norm of the operator, not by |alpha| itself, which may vanish.
        if abs(alpha.imag) > 1e-10 * lanczos_scale:
            raise RuntimeError(
                f"Lanczos diagonal coefficient alpha[{j}] = {alpha} has a "
                "significant imaginary part."
            )

        alpha = float(alpha.real)
        z -= alpha * q

        if reorthogonalize:
            for _ in range(2):
                for q_old in basis:
                    z -= hs_inner_product(q_old, z) * q_old

        alphas.append(alpha)

        if j == m - 1:
            break

        beta = hs_norm(z)
        if beta <= tol * lanczos_scale:      # invariant subspace reached
            break

        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    return (
        np.asarray(basis),
        np.asarray(alphas, dtype=float),
        np.asarray(betas, dtype=float),
        norm_opt,
    )


def krylov_modular(basis, alphas, betas, norm_opt, x, m=None, shift=0.0):
    r"""
    exp(-shift) * exp(x ad_H)(opt) restricted to the first m Krylov vectors.

    T is real symmetric tridiagonal, so expm(x T) e_1 is obtained from a
    dense symmetric eigendecomposition -- faster and better conditioned
    than a general expm, and it lets the scalar `shift` be applied inside
    the exponential where it cannot overflow.
    """
    k = len(alphas) if m is None else min(int(m), len(alphas))
    if k == 0:
        return np.zeros(basis.shape[1:], dtype=complex)

    T = np.diag(alphas[:k])
    if k > 1:
        off = betas[: k - 1]
        T += np.diag(off, 1) + np.diag(off, -1)

    w, S = la.eigh(T)
    # expm(x T) e_1 = S diag(exp(x w)) S^T e_1
    coefficients = S @ (np.exp(x * w - shift) * S[0, :])

    return norm_opt * np.tensordot(coefficients, basis[:k], axes=(0, 0))


# ----------------------------------------------------------------------
# Exact reference
# ----------------------------------------------------------------------

def exact_modular(eigenvalues, eigenvectors, opt_energy, x, shift=0.0):
    r"""
    exp(-shift) * exp(x ad_H)(opt), from the eigendecomposition of H.

    `opt_energy` = U^dag opt U is passed in precomputed because it does not
    depend on x.  With shift = |x| * (lambda_max - lambda_min) every exponent
    is <= 0, so this cannot overflow; the smallest entries underflow to
    zero, which is harmless at the 1e-300 relative level.
    """
    exponent = x * (eigenvalues[:, None] - eigenvalues[None, :]) - shift
    return eigenvectors @ (np.exp(exponent) * opt_energy) @ eigenvectors.conj().T


# ----------------------------------------------------------------------
# MH = sum_a Ga^dag Ga  and its low-lying spectrum
# ----------------------------------------------------------------------

def build_MH(L_stack, R_stack):
    r"""
    Assemble  MH = sum_a Ga^dag Ga  with  Ga = La (x) I - I (x) Ra^T,
    from stacks of shape (n_J, d, d), using only d x d products:

        MH = (sum_a La^dag La) (x) I
           + I (x) (sum_a conj(Ra) Ra^T)
           - D - D^dag,        D = sum_a La^dag (x) Ra^T.
    """
    d = L_stack.shape[1]
    Id = np.eye(d)

    A = np.einsum("aki,akj->ij", L_stack.conj(), L_stack)   # sum La^dag La
    C = np.einsum("aik,ajk->ij", R_stack.conj(), R_stack)   # sum (Ra^T)^dag Ra^T

    # D[i*d+k, j*d+l] = sum_a (La^dag)[i,j] (Ra^T)[k,l]
    D = np.einsum(
        "aij,akl->ikjl", L_stack.conj().transpose(0, 2, 1), R_stack.transpose(0, 2, 1)
    ).reshape(d * d, d * d)

    MH = np.kron(A, Id) + np.kron(Id, C)
    MH -= D
    MH -= D.conj().T
    return MH


def spectral_gap(MH, label=""):
    """lambda_1 - lambda_0 of MH / 2, plus a resolvability check.

    subset_by_index selects the two lowest eigenvalues only (zheevr), and
    eigvals_only skips the eigenvectors entirely.

    MH is positive semidefinite with lambda_0 ~ 0, so eigenvalues carry an
    absolute error of order eps * ||MH||.  A gap below ~1e-13 ||MH|| is
    therefore numerical noise, not physics; this is a property of the
    problem, not of the rescaling, since gap / ||MH|| is scale invariant.
    """
    vals = la.eigvalsh(MH, subset_by_index=[0, 1])
    gap = 0.5 * float(vals[1] - vals[0])

    scale = float(np.abs(np.diag(MH)).max())     # cheap proxy for ||MH||
    if scale > 0 and gap < 1e-13 * scale:
        print(
            f"    [warning] {label}gap/||MH|| = {gap / scale:.2e} is at the "
            "float64 noise floor",
            flush=True,
        )
    return gap


# ========== 全局固定参数 ==========
n = 6
d = int(2 ** n)
n2 = int(n * 2)
d2 = int(4 ** n)

# 加载哈密顿量（只加载一次，不用重复读文件）
H0 = np.load("FF_ham.npy")
V = np.load("Int_ham.npy")

m_list = np.array([12, 24, 36])
m_max = int(m_list.max())
n_col = 1 + len(m_list)          # 第0列精确，其余对应m_list

# 构造J算子集合（只构造一次）
SetofJ = []
for a in range(n):
    SetofJ.append(mycode.SingleX(a, n))
    SetofJ.append(mycode.SingleZ(a, n))
n_J = len(SetofJ)

L = 20
Listofbeta = np.linspace(1 / L, 8, L)

# 四个不同h，循环分别计算并保存
h_list = np.array([0.01, 0.1, 1, 10])

for h in h_list:
    print(f"===== 当前计算 h = {h} =====", flush=True)

    log10_gap = np.zeros((L, n_col))

    FH_ham = H0 + h * V
    eig, vec = la.eigh(FH_ham)
    width = float(eig[-1] - eig[0])          # = ||ad_H||, sets the shift

    # 每个J只做一次Lanczos（与x无关），并预先转到能量本征基
    krylov_data = [lanczos_tridiag(m_max, FH_ham, Ja) for Ja in SetofJ]
    opt_energy = [vec.conj().T @ Ja @ vec for Ja in SetofJ]

    for l, beta in enumerate(Listofbeta):
        x = 0.25 * beta
        shift = x * width
        # MH是按exp(-2*shift)的单位算出来的，间隙不是尺度不变量，
        # 所以最后要把这个因子加回去（在log空间里做，避免溢出）。
        log10_factor = 2.0 * shift / np.log(10.0)

        # ---- 精确 ----
        L_exact = np.array(
            [exact_modular(eig, vec, oe, -x, shift=shift) for oe in opt_energy]
        )
        R_exact = np.array(
            [exact_modular(eig, vec, oe, +x, shift=shift) for oe in opt_energy]
        )
        gap = spectral_gap(build_MH(L_exact, R_exact), label="exact ")
        log10_gap[l, 0] = np.log10(gap) + log10_factor if gap > 0 else -np.inf

        # ---- 三个不同m的Krylov近似 ----
        for i, m in enumerate(m_list):
            L_kry = np.array(
                [krylov_modular(*kd, -x, m=int(m), shift=shift) for kd in krylov_data]
            )
            R_kry = np.array(
                [krylov_modular(*kd, +x, m=int(m), shift=shift) for kd in krylov_data]
            )
            gap_k = spectral_gap(build_MH(L_kry, R_kry), label=f"m={m} ")
            log10_gap[l, i + 1] = (
                np.log10(gap_k) + log10_factor if gap_k > 0 else -np.inf
            )

        print(
            f"  beta = {beta:6.3f}  log10(gap) = "
            + "  ".join(
                f"{name}: {g:+.4f}"
                for name, g in zip(["exact"] + [f"m={m}" for m in m_list], log10_gap[l])
            ),
            flush=True,
        )

    # 原单位下的间隙；beta很大时可能溢出为inf，此时请使用log10文件
    with np.errstate(over="ignore"):
        Listofspectralgap = np.power(10.0, log10_gap)

    save_name = f"Listofspectralgap_h{h}.npy"
    np.save(save_name, Listofspectralgap)
    np.save(f"Listoflog10gap_h{h}.npy", log10_gap)

    n_inf = int(np.isinf(Listofspectralgap).sum())
    if n_inf:
        print(f"注意: {n_inf} 个数值超出float64范围，已在npy中记为inf")
    print(f"已保存 h={h} 数据至: {save_name}\n", flush=True)