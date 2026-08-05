"""
Krylov (Lanczos) approximation of the modular flow

    Delta^{s}(J) = exp(x ad_H)(J),   ad_H(X) = H X - X H,   x = beta/4,

benchmarked against the exact eigendecomposition result.

Key structural points
---------------------
* The Krylov space K_m(ad_H, J) does NOT depend on x.  The Lanczos run is
  therefore performed once per (H, J) pair at m = max(m_list), and every
  (beta, m) pair is obtained by exponentiating a small tridiagonal matrix.
  This replaces  n_beta * n_m  Lanczos runs by a single one.
* Lanczos is nested: the leading k x k block of T from an m-step run equals
  the T of a k-step run (exactly so with full reorthogonalization, since the
  recurrence at step j only ever touches basis vectors 0..j).
* Everything is evaluated with the global factor exp(-x * W) pulled out,
  W = lambda_max - lambda_min being the spectral width of H.  Both the exact
  and the Krylov result carry the same factor, so the *relative* error is
  unaffected while the intermediate matrices stay O(1) instead of
  overflowing float64 at large beta.
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

    Parameters
    ----------
    m : int
        Maximum Krylov dimension.
    ham : ndarray (d, d)
        Hermitian Hamiltonian.
    opt : ndarray (d, d)
        Starting operator.
    tol : float
        Relative breakdown tolerance for the off-diagonal coefficients.
    reorthogonalize : bool
        Two-pass full reorthogonalization against the stored basis.

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
    depend on x.  With shift = x * (lambda_max - lambda_min) every exponent
    is <= 0, so this cannot overflow; the smallest entries underflow to
    zero, which is harmless at the 1e-300 relative level.
    """
    exponent = x * (eigenvalues[:, None] - eigenvalues[None, :]) - shift
    return eigenvectors @ (np.exp(exponent) * opt_energy) @ eigenvectors.conj().T


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------

def main():
    n = 6
    d = 2 ** n

    H0 = np.load("FF_ham.npy")
    V = np.load("Int_ham.npy")

    m_list = np.array([12, 24, 36])
    m_max = int(m_list.max())

    # Local Pauli operators X_a, Z_a.
    SetofJ = []
    for a in range(n):
        SetofJ.append(mycode.SingleX(a, n))
        SetofJ.append(mycode.SingleZ(a, n))
    n_J = len(SetofJ)

    L = 20
    Listofbeta = np.linspace(1.0 / L, 8.0, L)
    h_list = np.array([0.01, 0.1, 1.0, 10.0])

    for h in h_list:
        print(f"===== h = {h} =====", flush=True)

        FH_ham = H0 + h * V
        eig, vec = la.eigh(FH_ham)
        width = float(eig[-1] - eig[0])       # spectral width -> ||ad_H||

        # --- one Lanczos run per operator, reused for every (beta, m) ---
        krylov_data = []
        opt_energy = []
        for Ja in SetofJ:
            krylov_data.append(lanczos_tridiag(m_max, FH_ham, Ja))
            opt_energy.append(vec.conj().T @ Ja @ vec)

        rel_err = np.zeros((L, len(m_list)))
        log10_scale = np.zeros((L, len(m_list)))   # log10 of the exact norm

        for l, beta in enumerate(Listofbeta):
            x = 0.25 * beta
            shift = x * width                  # keeps both sides O(1)

            err_tmp = np.zeros((len(m_list), n_J))
            scale_tmp = np.zeros((len(m_list), n_J))

            for a in range(n_J):
                exact = exact_modular(eig, vec, opt_energy[a], x, shift=shift)
                norm_exact = matrix_norm(exact)

                basis, alphas, betas, norm_opt = krylov_data[a]

                for i, m in enumerate(m_list):
                    approx = krylov_modular(
                        basis, alphas, betas, norm_opt, x, m=int(m), shift=shift
                    )
                    err = matrix_norm(exact - approx)
                    err_tmp[i, a] = err / norm_exact if norm_exact > 0 else np.nan
                    scale_tmp[i, a] = norm_exact

            # Worst case over the 2n local operators.
            worst = np.argmax(err_tmp, axis=1)
            rel_err[l, :] = err_tmp[np.arange(len(m_list)), worst]
            # Absolute error = rel_err * 10**log10_scale.
            log10_scale[l, :] = (
                np.log10(scale_tmp[np.arange(len(m_list)), worst])
                + shift / np.log(10.0)
            )

            print(
                f"  beta = {beta:6.3f}   rel. err = "
                + "  ".join(f"m={m}: {e:.3e}" for m, e in zip(m_list, rel_err[l]))
                , flush=True
            )

        # Relative error, shape (L, len(m_list)) -- same layout as before.
        out = f"Listofnorm_h{h}.npy"
        np.save(out, rel_err)

        # log10 ||exact||_2 for the same worst-case operator, so that
        # absolute error = rel_err * 10**log10_norm_exact.
        # out_scale = f"Lognorm_h{h}.npy"
        # np.save(out_scale, log10_scale)

        # print(f"saved -> {out}, {out_scale}\n", flush=True)


if __name__ == "__main__":
    main()
