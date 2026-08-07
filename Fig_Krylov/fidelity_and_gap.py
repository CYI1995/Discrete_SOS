"""
Krylov parent-Hamiltonian ground-state fidelity -- computation only.

H0 and V are read from FF_ham.npy and Int_ham.npy; the generators {X_i, Z_i}
are constructed internally.  The results are written to

    data_fidelity_and_gap.npy

as a single pickled dict, to be read back with

    results = np.load("data_fidelity_and_gap.npy", allow_pickle=True).item()

No figure is produced here; plotting is a separate script.

Output format
-------------
results is keyed by float(h).  For each h, with n = number of beta values
that passed every check (0 <= n <= N_BETA):

    results[h]["beta"]          (n,)               the beta actually computed
    results[h]["fidelity"]      (n, len(M_LIST))   column i <-> M_LIST[i]
    results[h]["absolute_gap"]  (n, len(M_LIST))   unshifted absolute gap
    results[h]["relative_gap"]  (n, len(M_LIST))   diagnostic only
    results[h]["sin_theta"]     (n, len(M_LIST))   same column convention

Because the sweep stops at the first unresolvable beta, n is generally
smaller than N_BETA and the stored "beta" array is a prefix of BETA_GRID.
Plotting code must use results[h]["beta"] as the abscissa rather than
rebuilding the full grid.

Model
-----
    H(h) = H0 + h V,

with H0 and V supplied by the input files.  Both are checked to be square of
size 2**N_QUBITS and Hermitian to round-off before use.

IMPORTANT -- convention matching.  The generators built here place site 0 in
the leftmost tensor factor, i.e. X_2 = I (x) I (x) X (x) I (x) I (x) I.  If
FF_ham.npy was produced with the opposite site ordering, H and J_a live in
mirrored conventions and every result below is silently wrong.  Check once
against the code that generated the files, e.g.

    assert np.allclose(build_generators(6)[0], mycode.SingleX(0, 6))

What is computed
----------------
    M_H = sum_a Gamma_a^dag Gamma_a,   Gamma_a = L_a (x) I - I (x) R_a^T,
    L_a = exp(-x ad_H)(J_a),  R_a = exp(+x ad_H)(J_a),  x = beta/4.

M_H is positive semidefinite and its kernel is one-dimensional: M_H X = 0 iff
[J_a, Y] = 0 for all a with X = e^{-xH} Y e^{-xH}; since {X_i, Z_i} generates
the full matrix algebra, Y is proportional to the identity and X is
proportional to exp(-beta H/2), the unnormalized square root of the Gibbs
state.  vec(X) is the purified Gibbs vector.  The exact ground state is
therefore written down in closed form, not diagonalized, and the quantity
stored is |<gs_exact, gs_krylov>|^2 for the Krylov-truncated M_H.

Simplifications relied on here
------------------------------
1.  K_m(ad_H, J_a) depends only on (H, J_a), not on x or its sign.  One
    Lanczos run per generator at max(M_LIST) serves L_a, R_a, every beta and
    every m; smaller m are nested prefixes of the same tridiagonal T.

2.  R_a = L_a^dag exactly, on the exact and the truncated side alike.  J_a is
    Hermitian, so ad_H alternates Hermitian and anti-Hermitian, every Lanczos
    coefficient alpha_k vanishes, T anticommutes with diag((-1)^k), and the
    two sign flips cancel.  Only L_a is built; the assertion on alpha_max
    guards the assumption.  This also makes the second Kronecker term
    conj(A) = A^T, so build_parent_hamiltonian needs one einsum fewer.

3.  The global factor exp(-x W), W = lambda_max - lambda_min, is pulled out of
    L_a and R_a.  Since the spectrum of ad_H lies in [-W, W], every exponent
    is <= 0 and overflow is impossible; M_H is rescaled by exp(-2 x W), which
    leaves its eigenvectors untouched.  Underflow of the smallest entries is
    harmless.

Truncation criterion
--------------------
The sweep is truncated when the ABSOLUTE gap of the unshifted parent
Hamiltonian falls to ABS_GAP_TOL.

The matrix that is diagonalized is the shifted parent

    M_shifted = exp(-2 x W) M_unshifted.

Hence the two gaps satisfy

    gap_unshifted = exp(2 x W) gap_shifted.

The comparison with ABS_GAP_TOL is performed in the logarithmic domain, so the
undoing of the global shift does not overflow.  The stored "absolute_gap" is
gap_unshifted.  The relative gap is still stored as a diagnostic but does not
control truncation.

The Davis--Kahan quantity sin_theta is retained only as a diagnostic.  Apart
from a nonpositive/nonfinite eigensolver gap, the sweep is truncated only when
absolute_gap <= ABS_GAP_TOL.
"""

from pathlib import Path
import math
import numpy as np
import scipy.linalg as la


# ==============================================================
# Parameters
# ==============================================================

H0 = np.load('FF_ham.npy')
interaction = np.load('Int_ham.npy')

M_LIST = (8, 16, 24)
H_LIST = (0.01, 0.1, 1.0, 10.0)

N_BETA = 20
BETA_GRID = np.linspace(1.0 / 20.0, 8.0, N_BETA)

FF_HAM_PATH = Path("FF_ham.npy")     # H0
INT_HAM_PATH = Path("Int_ham.npy")   # V

N_QUBITS = 4

RESULTS_PATH = Path("data_fidelity_and_gap.npy")

ABS_GAP_TOL = 1e-8      # cutoff for the unshifted absolute parent-Hamiltonian gap
ANGLE_TOL = 1e-4          # warning threshold only; it does not truncate
LANCZOS_TOL = 1e-13
ALPHA_TOL = 1e-10       # guards R_a = L_a^dag
HERMITICITY_TOL = 1e-10 # relative anti-Hermitian part allowed in the inputs


# ==============================================================
# Pauli matrices and tensor-product helpers
# ==============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(factors):
    """Kronecker product of a list of matrices."""
    result = np.ones((1, 1), dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def one_site(operator, site, number_of_qubits):
    """Embed one operator at `site`; site 0 is the leftmost tensor factor."""
    if not 0 <= site < number_of_qubits:
        raise ValueError(f"Invalid site {site} for {number_of_qubits} qubits.")
    factors = [I2] * number_of_qubits
    factors[site] = operator
    return kron_all(factors)


def build_generators(number_of_qubits):
    """The irreducible family {X_i, Z_i}."""
    return [
        one_site(operator, site, number_of_qubits)
        for site in range(number_of_qubits)
        for operator in (X, Z)
    ]


def hermitian_part(M):
    """Drop the roundoff-level anti-Hermitian part."""
    return 0.5 * (M + M.conj().T)


def hs_inner_product(A, B):
    """Hilbert-Schmidt inner product Tr(A^dag B)."""
    return np.vdot(A, B)


def hs_norm(A):
    """Hilbert-Schmidt / Frobenius norm."""
    return la.norm(A, ord="fro")


# ==============================================================
# Lanczos for ad_H, and the modular flow in the Krylov space
# ==============================================================

def lanczos_tridiag(m, hamiltonian, operator, tolerance=LANCZOS_TOL):
    """
    Lanczos for ad_H(A) = H A - A H under the Hilbert-Schmidt inner product.

    Returns (basis, alphas, betas, norm_operator).  The leading k x k block of
    the tridiagonal matrix equals that of a k-step run, so one run at
    max(M_LIST) supplies every smaller Krylov order.
    """
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12):
        raise ValueError("ad_H is self-adjoint only for Hermitian H.")

    norm_operator = hs_norm(operator)
    if norm_operator == 0.0:
        raise ValueError("The starting operator must be nonzero.")

    # ||ad_H|| <= 2 ||H||_2 sets the scale for every coefficient.
    scale = max(1.0, 2.0 * la.norm(hamiltonian, 2))

    q = operator / norm_operator
    q_previous = np.zeros_like(q)
    beta_previous = 0.0

    basis, alphas, betas = [], [], []

    for step in range(int(m)):
        basis.append(q.copy())

        z = hamiltonian @ q - q @ hamiltonian
        if step > 0:
            z -= beta_previous * q_previous

        alpha = hs_inner_product(q, z)
        if abs(alpha.imag) > 1e-10 * scale:
            raise RuntimeError(f"alpha[{step}] has a significant imaginary part.")
        alpha = float(alpha.real)
        z -= alpha * q

        for _ in range(2):                      # full reorthogonalization
            for old_vector in basis:
                z -= hs_inner_product(old_vector, z) * old_vector

        alphas.append(alpha)

        if step == int(m) - 1:
            break

        beta_value = float(hs_norm(z))
        if beta_value <= tolerance * scale:     # invariant subspace reached
            break

        betas.append(beta_value)
        q_previous, q = q, z / beta_value
        beta_previous = beta_value

    return (
        np.asarray(basis, dtype=complex),
        np.asarray(alphas, dtype=float),
        np.asarray(betas, dtype=float),
        float(norm_operator),
    )


def krylov_modular(basis, alphas, betas, norm_operator, x, m, shift):
    """
    exp(-shift) exp(x ad_H)(operator) restricted to the first m Krylov vectors.

    T is real symmetric tridiagonal, so exp(x T) e_1 comes from a dense
    symmetric eigendecomposition, and the shift is applied inside the
    exponential.  With shift = |x| W every exponent is <= 0, so np.exp cannot
    overflow here.
    """
    k = min(int(m), len(alphas))
    tridiagonal = np.diag(alphas[:k])
    if k > 1:
        off_diagonal = betas[: k - 1]
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    ritz_values, ritz_vectors = la.eigh(tridiagonal, check_finite=False)
    coefficients = ritz_vectors @ (
        np.exp(x * ritz_values - shift) * ritz_vectors[0, :]
    )

    return norm_operator * np.tensordot(coefficients, basis[:k], axes=(0, 0))


# ==============================================================
# Parent Hamiltonian
# ==============================================================

def build_parent_hamiltonian(left_stack):
    r"""
    M_H = sum_a Gamma_a^dag Gamma_a with Gamma_a = L_a (x) I - I (x) R_a^T and
    R_a = L_a^dag, expanded so that only d x d products are needed:

        M_H = A (x) I + I (x) A^T - D - D^dag,
        A   = sum_a L_a^dag L_a,   D = sum_a L_a^dag (x) conj(L_a).

    The kron(A, I) convention means M_H acts on vec(X) = X.reshape(-1) in row
    major order, i.e. Gamma_a vec(X) = vec(L_a X - X R_a).

    D is one GEMM: the equivalent einsum outer product does not reach BLAS.
    """
    number_of_generators, dimension = left_stack.shape[:2]
    identity = np.eye(dimension, dtype=complex)

    A = np.einsum("aki,akj->ij", left_stack.conj(), left_stack, optimize=True)

    # Ld[a, i*d+j] = (L_a^dag)[i,j],  Rt[a, k*d+l] = (R_a^T)[k,l] = conj(L_a)
    flat = (number_of_generators, dimension * dimension)
    Ld = left_stack.conj().transpose(0, 2, 1).reshape(flat)
    Rt = left_stack.conj().reshape(flat)

    # (Ld.T @ Rt)[i,j,k,l] -> D[i*d+k, j*d+l]
    D = (Ld.T @ Rt).reshape((dimension,) * 4).transpose(0, 2, 1, 3)
    D = D.reshape(dimension * dimension, dimension * dimension)

    parent = np.kron(A, identity) + np.kron(identity, A.T)
    parent -= D
    parent -= D.conj().T
    return 0.5*hermitian_part(parent)


def exact_purified_gibbs_vector(energies, eigenvectors, beta):
    """
    Normalized row-vectorization of exp(-beta H / 2).

    Subtracting energies[0] is an overall constant that cancels in the
    normalization; it makes every exponent <= 0, so only underflow is
    possible and weights[0] = 1 keeps the norm away from zero.
    """
    weights = np.exp(-0.5 * beta * (energies - energies[0]))
    root_gibbs = (eigenvectors * weights[None, :]) @ eigenvectors.conj().T
    vector = root_gibbs.reshape(-1)
    return vector / la.norm(vector)


def checked_ground_state(parent, shift):
    """
    Lowest eigenpair of the shifted parent Hamiltonian, or None when it is not
    numerically resolvable.

    The diagonalized matrix is

        M_shifted = exp(-2 * shift) M_unshifted,

    so

        absolute_gap = exp(2 * shift) * shifted_gap.

    The absolute-gap cutoff is evaluated in the logarithmic domain to avoid
    overflow when undoing the global shift.

        relative_gap = shifted_gap / ||M_shifted||_inf
        sin_theta    = ||M_shifted gs - lambda_0 gs|| / shifted_gap

    The relative gap and sin_theta are stored only as diagnostics.  Apart from
    a nonpositive/nonfinite eigensolver gap, truncation is controlled only by
    ABS_GAP_TOL.
    """
    eigenvalues, eigenvectors = la.eigh(
        parent, subset_by_index=[0, 1], check_finite=False
    )
    lambda_0, lambda_1 = float(eigenvalues[0]), float(eigenvalues[1])
    shifted_gap = lambda_1 - lambda_0

    if not np.isfinite(shifted_gap) or shifted_gap <= 0.0:
        return None

    # Compare the unshifted absolute gap in log space:
    # log(gap_unshifted) = log(gap_shifted) + 2 * shift.
    log_absolute_gap = float(np.log(shifted_gap) + 2.0 * shift)
    log_abs_gap_tol = float(np.log(ABS_GAP_TOL))

    # Store the ordinary gap whenever representable.  If it overflows, it is
    # certainly above the lower cutoff, so +inf is an honest diagnostic value.
    if log_absolute_gap > np.log(np.finfo(float).max):
        absolute_gap = float("inf")
    else:
        absolute_gap = float(np.exp(log_absolute_gap))

    scale = float(la.norm(parent, ord=np.inf))
    if not np.isfinite(scale) or scale <= 0.0:
        return None

    relative_gap = shifted_gap / scale

    ground_state = eigenvectors[:, 0]
    residual = float(la.norm(parent @ ground_state - lambda_0 * ground_state))
    sin_theta = residual / shifted_gap

    if (
        not np.isfinite(relative_gap)
        or not np.isfinite(residual)
        or not np.isfinite(sin_theta)
    ):
        return None

    if log_absolute_gap <= log_abs_gap_tol:
        return None

    return {
        "absolute_gap": absolute_gap,
        "shifted_gap": shifted_gap,
        "relative_gap": relative_gap,
        "sin_theta": sin_theta,
        "ground_state": ground_state,
    }


# ==============================================================
# Computation
# ==============================================================

def compute_results():
    """Compute every curve; see the module docstring for the output layout."""
    print("Loading Hamiltonians:", flush=True)
    generators = build_generators(N_QUBITS)
    maximum_order = max(M_LIST)

    results = {}

    for h in H_LIST:
        print(f"\n===== h = {h:g} =====", flush=True)

        hamiltonian = hermitian_part(H0 + h * interaction)
        energies, eigenvectors = la.eigh(hamiltonian)
        width = float(energies[-1] - energies[0])

        krylov_data = [
            lanczos_tridiag(maximum_order, hamiltonian, generator)
            for generator in generators
        ]

        # R_a = L_a^dag requires every Lanczos diagonal coefficient to vanish.
        alpha_max = max(float(np.abs(a).max()) for _, a, _, _ in krylov_data)
        alpha_scale = max(1.0, 2.0 * la.norm(hamiltonian, 2))
        if alpha_max > ALPHA_TOL * alpha_scale:
            raise RuntimeError(
                f"max|alpha| = {alpha_max:.3e} exceeds tolerance; "
                "R_a = L_a^dag no longer holds."
            )

        print(
            f"  W = {width:.4f}, ABS_GAP_TOL = {ABS_GAP_TOL:.1e}",
            flush=True,
        )

        valid_beta = []
        fidelity_rows = []
        absolute_gap_rows = []
        relative_gap_rows = []
        sin_theta_rows = []

        for beta in BETA_GRID:
            x = 0.25 * float(beta)
            shift = x * width       # rescales M_H only; eigenvectors unchanged

            exact_ground_state = exact_purified_gibbs_vector(
                energies, eigenvectors, float(beta)
            )

            fidelity_row = []
            absolute_gap_row = []
            relative_gap_row = []
            sin_theta_row = []
            beta_is_valid = True

            for m in M_LIST:
                left_stack = np.asarray(
                    [
                        krylov_modular(*data, x=-x, m=m, shift=shift)
                        for data in krylov_data
                    ]
                )
                parent = build_parent_hamiltonian(left_stack)
                del left_stack

                diagnostic = checked_ground_state(parent, shift)
                del parent

                if diagnostic is None:
                    print(
                        f"  beta={beta:.6f}, m={m}: unshifted absolute gap "
                        f"<= {ABS_GAP_TOL:.1e}, or the computed gap is "
                        "nonpositive/nonfinite; truncating this h curve.",
                        flush=True,
                    )
                    beta_is_valid = False
                    break

                if diagnostic["sin_theta"] > ANGLE_TOL:
                    print(
                        f"  warning: beta={beta:.6f}, m={m}, "
                        f"sin_theta={diagnostic['sin_theta']:.3e} "
                        f"> {ANGLE_TOL:.1e}; point retained because "
                        "ABS_GAP is the sole truncation criterion.",
                        flush=True,
                    )

                overlap = np.vdot(exact_ground_state, diagnostic["ground_state"])
                fidelity_row.append(float(np.clip(abs(overlap) ** 2, 0.0, 1.0)))
                absolute_gap_row.append(diagnostic["absolute_gap"])
                relative_gap_row.append(diagnostic["relative_gap"])
                sin_theta_row.append(diagnostic["sin_theta"])

            if not beta_is_valid:
                break

            valid_beta.append(float(beta))
            fidelity_rows.append(fidelity_row)
            absolute_gap_rows.append(absolute_gap_row)
            relative_gap_rows.append(relative_gap_row)
            sin_theta_rows.append(sin_theta_row)

            print(
                f"  beta={beta:7.4f}  "
                + "  ".join(
                    f"m={m}: {value:.8f}"
                    for m, value in zip(M_LIST, fidelity_row)
                )
                + f"   abs_gap={absolute_gap_row[0]:.2e}"
                + f"   rel_gap={relative_gap_row[0]:.2e}",
                flush=True,
            )

        results[float(h)] = {
            "beta": np.asarray(valid_beta, dtype=float),
            "fidelity": np.asarray(fidelity_rows, dtype=float).reshape(
                -1, len(M_LIST)
            ),
            "absolute_gap": np.asarray(absolute_gap_rows, dtype=float).reshape(
                -1, len(M_LIST)
            ),
            "relative_gap": np.asarray(relative_gap_rows, dtype=float).reshape(
                -1, len(M_LIST)
            ),
            "sin_theta": np.asarray(sin_theta_rows, dtype=float).reshape(
                -1, len(M_LIST)
            ),
        }

    return results


def save_results(results, path=RESULTS_PATH):
    """
    Write the results dict to `path`.

    np.save wraps the dict in a 0-d object array, so reading it back needs
    allow_pickle=True and .item().  The metadata entry records the settings
    the numbers were produced with, so a stale file can be spotted.
    """
    payload = dict(results)
    payload["metadata"] = {
        "n_qubits": N_QUBITS,
        "m_list": np.asarray(M_LIST, dtype=int),
        "h_list": np.asarray(H_LIST, dtype=float),
        "beta_grid": BETA_GRID,
        "abs_gap_tol": ABS_GAP_TOL,
        "angle_tol": ANGLE_TOL,
        "absolute_gap_convention": (
            "gap of the unshifted parent Hamiltonian; "
            "gap_unshifted = exp(2*x*W) * gap_shifted"
        ),
    }
    np.save(path, payload, allow_pickle=True)
    return path


# ==============================================================
# Main
# ==============================================================

def main():
    results = compute_results()
    path = save_results(results)

    print(f"\nSaved results to {path.resolve()}", flush=True)
    print(
        "Read back with: "
        "np.load(path, allow_pickle=True).item()",
        flush=True,
    )

    print("\nValid beta points:", flush=True)
    for h in H_LIST:
        beta = results[float(h)]["beta"]
        if beta.size == 0:
            status = "none"
        elif beta.size == N_BETA:
            status = f"{beta.size}/{N_BETA}, complete grid"
        else:
            status = f"{beta.size}/{N_BETA}, last beta={beta[-1]:.6f}"
        print(f"  h={h:g}: {status}", flush=True)


if __name__ == "__main__":
    main()