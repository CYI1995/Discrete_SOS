"""
Exact parent-Hamiltonian spectral gap -- computation only.

Inputs
------
    FF_ham.npy    : H0
    Int_ham.npy   : V

Model
-----
    H(h) = H0 + h V.

For each h and beta, this script constructs the exact modular operators

    L_a = exp(-x H) J_a exp(+x H) = exp(-x ad_H)(J_a),
    R_a = L_a^dagger,
    x = beta / 4,

and the exact parent Hamiltonian

    M_H = sum_a Gamma_a^dagger Gamma_a,
    Gamma_a = L_a tensor I - I tensor R_a^T.

To avoid overflow, each L_a is multiplied by exp(-x W), where

    W = lambda_max(H) - lambda_min(H).

Thus the matrix actually diagonalized is

    M_shifted = exp(-2 x W) M_exact.

The exact unshifted spectral gap is recovered from

    gap_exact = exp(2 x W) gap_shifted.

The results are saved in exact_gap.npy as a pickled dictionary. Read it with

    results = np.load("exact_gap.npy", allow_pickle=True).item()

Output format
-------------
For every h, with n valid beta values,

    results[h]["beta"]                   shape (n,)
    results[h]["absolute_gap"]           shape (n,)
    results[h]["log_absolute_gap"]       shape (n,)
    results[h]["shifted_gap"]            shape (n,)
    results[h]["relative_gap"]           shape (n,)
    results[h]["ground_energy_shifted"]  shape (n,)
    results[h]["ground_residual"]        shape (n,)

The sweep stops at the first beta for which the unshifted exact gap satisfies

    absolute_gap <= ABS_GAP_TOL,

or the two lowest eigenvalues are not numerically resolvable.

No Krylov approximation and no fidelity calculation are used here.
"""

from pathlib import Path

import numpy as np
import scipy.linalg as la


# ==============================================================
# Parameters
# ==============================================================

N_QUBITS = 4
H_LIST = (0.01, 0.1, 1.0, 10.0)

N_BETA = 20
BETA_GRID = np.linspace(1.0 / 20.0, 8.0, N_BETA)

FF_HAM_PATH = Path("FF_ham.npy")
INT_HAM_PATH = Path("Int_ham.npy")
RESULTS_PATH = Path("exact_gap.npy")

ABS_GAP_TOL = 1e-8
HERMITICITY_TOL = 1e-10
GROUND_RESIDUAL_WARN = 1e-8


# ==============================================================
# Pauli generators {X_i, Z_i}
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
    """Embed one operator at site; site 0 is the leftmost tensor factor."""
    if not 0 <= site < number_of_qubits:
        raise ValueError(f"Invalid site {site} for {number_of_qubits} qubits.")

    factors = [I2] * number_of_qubits
    factors[site] = operator
    return kron_all(factors)


def build_generators(number_of_qubits):
    """Construct the irreducible family {X_i, Z_i}."""
    return [
        one_site(operator, site, number_of_qubits)
        for site in range(number_of_qubits)
        for operator in (X, Z)
    ]


# ==============================================================
# Input and basic helpers
# ==============================================================


def hermitian_part(matrix):
    """Remove the roundoff-level anti-Hermitian component."""
    return 0.5 * (matrix + matrix.conj().T)


def load_model_hamiltonians(number_of_qubits):
    """Load and validate H0 and V."""
    dimension = 2 ** number_of_qubits
    loaded = {}

    for name, path in (("H0", FF_HAM_PATH), ("V", INT_HAM_PATH)):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} was not found in {path.resolve().parent}."
            )

        matrix = np.asarray(np.load(path), dtype=complex)

        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"{path}: shape {matrix.shape} != {(dimension, dimension)}; "
                f"N_QUBITS={number_of_qubits} does not match the input file."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path} contains NaN or infinite values.")

        denominator = max(1.0, float(la.norm(matrix, ord="fro")))
        asymmetry = float(
            la.norm(matrix - matrix.conj().T, ord="fro") / denominator
        )

        if asymmetry > HERMITICITY_TOL:
            raise ValueError(
                f"{path}: relative anti-Hermitian part {asymmetry:.3e} "
                f"exceeds {HERMITICITY_TOL:.1e}."
            )

        loaded[name] = hermitian_part(matrix)
        print(
            f"  loaded {name} from {path} "
            f"({matrix.shape[0]}x{matrix.shape[1]})",
            flush=True,
        )

    return loaded["H0"], loaded["V"]


# ==============================================================
# Exact modular operators and exact parent Hamiltonian
# ==============================================================


def exact_left_modular(
    energies,
    eigenvectors,
    generator_in_energy_basis,
    x,
    shift,
):
    r"""
    Compute

        exp(-shift) exp(-x ad_H)(J)
        = exp(-shift) exp(-x H) J exp(+x H).

    With shift=xW, every scalar exponential has exponent <= 0.
    """
    energy_difference = energies[:, None] - energies[None, :]
    exponent = -x * energy_difference - shift

    transformed_energy_basis = (
        np.exp(exponent) * generator_in_energy_basis
    )

    return (
        eigenvectors
        @ transformed_energy_basis
        @ eigenvectors.conj().T
    )


def build_parent_hamiltonian(left_stack):
    r"""
    Build the shifted exact parent Hamiltonian

        M = sum_a Gamma_a^dagger Gamma_a,
        Gamma_a = L_a tensor I - I tensor R_a^T,
        R_a = L_a^dagger.

    For row-major vectorization,

        Gamma_a vec(X) = vec(L_a X - X R_a).
    """
    left_stack = np.asarray(left_stack, dtype=complex)

    if left_stack.ndim != 3:
        raise ValueError("left_stack must have shape (n_generators, d, d).")

    number_of_generators, dimension, second_dimension = left_stack.shape

    if number_of_generators == 0 or dimension != second_dimension:
        raise ValueError("left_stack must contain nonempty square matrices.")

    identity = np.eye(dimension, dtype=complex)

    # A = sum_a L_a^dagger L_a.
    A = np.einsum(
        "aki,akj->ij",
        left_stack.conj(),
        left_stack,
        optimize=True,
    )

    # D = sum_a L_a^dagger tensor conj(L_a).
    flat_shape = (number_of_generators, dimension * dimension)
    left_dagger_flat = (
        left_stack.conj().transpose(0, 2, 1).reshape(flat_shape)
    )
    conjugate_left_flat = left_stack.conj().reshape(flat_shape)

    D = (
        (left_dagger_flat.T @ conjugate_left_flat)
        .reshape((dimension,) * 4)
        .transpose(0, 2, 1, 3)
        .reshape(dimension * dimension, dimension * dimension)
    )

    parent = np.kron(A, identity) + np.kron(identity, A.T)
    parent -= D
    parent -= D.conj().T

    return hermitian_part(parent)


def exact_purified_gibbs_vector(energies, eigenvectors, beta):
    """Normalized row-vectorization of exp(-beta H/2)."""
    weights = np.exp(-0.5 * beta * (energies - energies[0]))
    root_gibbs = (
        (eigenvectors * weights[None, :])
        @ eigenvectors.conj().T
    )
    vector = root_gibbs.reshape(-1, order="C")
    norm = float(la.norm(vector))

    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Could not normalize the purified Gibbs vector.")

    return vector / norm


# ==============================================================
# Exact spectral-gap diagnostic
# ==============================================================


def exact_gap_diagnostic(parent_shifted, shift, exact_ground_state):
    r"""
    Compute the two lowest eigenvalues of

        M_shifted = exp(-2 shift) M_exact.

    The unshifted exact gap is

        gap_exact = exp(2 shift) gap_shifted.

    The ABS_GAP cutoff is evaluated in logarithmic form.
    """
    eigenvalues = la.eigvalsh(
        parent_shifted,
        subset_by_index=[0, 1],
        check_finite=False,
    )

    lambda_0 = float(eigenvalues[0])
    lambda_1 = float(eigenvalues[1])
    shifted_gap = lambda_1 - lambda_0

    if not np.isfinite(shifted_gap) or shifted_gap <= 0.0:
        return None, "shifted gap is nonpositive or nonfinite"

    scale = float(la.norm(parent_shifted, ord=np.inf))

    if not np.isfinite(scale) or scale <= 0.0:
        return None, "parent-Hamiltonian scale is nonpositive or nonfinite"

    relative_gap = shifted_gap / scale
    log_absolute_gap = float(np.log(shifted_gap) + 2.0 * shift)

    log_max_float = float(np.log(np.finfo(float).max))
    if log_absolute_gap > log_max_float:
        absolute_gap = float("inf")
    else:
        absolute_gap = float(np.exp(log_absolute_gap))

    residual = float(la.norm(parent_shifted @ exact_ground_state))
    relative_residual = residual / scale

    if not np.isfinite(relative_gap) or not np.isfinite(relative_residual):
        return None, "relative gap or ground-state residual is nonfinite"

    if log_absolute_gap <= float(np.log(ABS_GAP_TOL)):
        return None, (
            f"exact unshifted gap {absolute_gap:.3e} "
            f"<= ABS_GAP_TOL={ABS_GAP_TOL:.3e}"
        )

    return {
        "absolute_gap": absolute_gap,
        "log_absolute_gap": log_absolute_gap,
        "shifted_gap": shifted_gap,
        "relative_gap": relative_gap,
        "ground_energy_shifted": lambda_0,
        "ground_residual": relative_residual,
    }, None


# ==============================================================
# Computation and saving
# ==============================================================


def compute_results():
    """Compute the exact parent-Hamiltonian gap for every h and beta."""
    print("Loading Hamiltonians:", flush=True)
    H0, interaction = load_model_hamiltonians(N_QUBITS)
    generators = build_generators(N_QUBITS)

    results = {}

    for h in H_LIST:
        print(f"\n===== h = {h:g} =====", flush=True)

        hamiltonian = hermitian_part(H0 + h * interaction)
        energies, eigenvectors = la.eigh(hamiltonian, check_finite=True)
        width = float(energies[-1] - energies[0])

        generator_energy_basis = [
            eigenvectors.conj().T @ generator @ eigenvectors
            for generator in generators
        ]

        print(
            f"  W={width:.6e}, ABS_GAP_TOL={ABS_GAP_TOL:.1e}",
            flush=True,
        )

        valid_beta = []
        absolute_gaps = []
        log_absolute_gaps = []
        shifted_gaps = []
        relative_gaps = []
        ground_energies = []
        ground_residuals = []

        for beta in BETA_GRID:
            beta = float(beta)
            x = beta / 4.0
            shift = x * width

            left_stack = np.asarray(
                [
                    exact_left_modular(
                        energies,
                        eigenvectors,
                        generator_energy,
                        x,
                        shift,
                    )
                    for generator_energy in generator_energy_basis
                ],
                dtype=complex,
            )

            parent_shifted = build_parent_hamiltonian(left_stack)
            exact_ground_state = exact_purified_gibbs_vector(
                energies,
                eigenvectors,
                beta,
            )

            diagnostic, failure_reason = exact_gap_diagnostic(
                parent_shifted,
                shift,
                exact_ground_state,
            )

            del left_stack
            del parent_shifted

            if diagnostic is None:
                print(
                    f"  beta={beta:.6f}: {failure_reason}; "
                    "truncating this h curve.",
                    flush=True,
                )
                break

            if diagnostic["ground_residual"] > GROUND_RESIDUAL_WARN:
                print(
                    f"  warning: beta={beta:.6f}, relative exact-ground "
                    f"residual={diagnostic['ground_residual']:.3e}",
                    flush=True,
                )

            valid_beta.append(beta)
            absolute_gaps.append(diagnostic["absolute_gap"])
            log_absolute_gaps.append(diagnostic["log_absolute_gap"])
            shifted_gaps.append(diagnostic["shifted_gap"])
            relative_gaps.append(diagnostic["relative_gap"])
            ground_energies.append(diagnostic["ground_energy_shifted"])
            ground_residuals.append(diagnostic["ground_residual"])

            print(
                f"  beta={beta:7.4f}  "
                f"exact_gap={diagnostic['absolute_gap']:.8e}  "
                f"shifted_gap={diagnostic['shifted_gap']:.3e}  "
                f"relative_gap={diagnostic['relative_gap']:.3e}  "
                f"ground_res={diagnostic['ground_residual']:.3e}",
                flush=True,
            )

        results[float(h)] = {
            "beta": np.asarray(valid_beta, dtype=float),
            "absolute_gap": np.asarray(absolute_gaps, dtype=float),
            "log_absolute_gap": np.asarray(log_absolute_gaps, dtype=float),
            "shifted_gap": np.asarray(shifted_gaps, dtype=float),
            "relative_gap": np.asarray(relative_gaps, dtype=float),
            "ground_energy_shifted": np.asarray(ground_energies, dtype=float),
            "ground_residual": np.asarray(ground_residuals, dtype=float),
        }

    return results


def save_results(results, path=RESULTS_PATH):
    """Save the exact-gap results and metadata."""
    payload = dict(results)
    payload["metadata"] = {
        "n_qubits": N_QUBITS,
        "h_list": np.asarray(H_LIST, dtype=float),
        "beta_grid": np.asarray(BETA_GRID, dtype=float),
        "abs_gap_tol": ABS_GAP_TOL,
        "generator_family": "{X_i, Z_i}",
        "gap_convention": (
            "unshifted exact parent-Hamiltonian gap; "
            "gap_exact = exp(2*x*W) * gap_shifted"
        ),
    }

    np.save(path, payload, allow_pickle=True)
    return path


def main():
    results = compute_results()
    output_path = save_results(results)

    print(f"\nSaved exact gaps to {output_path.resolve()}", flush=True)
    print(
        'Read with: np.load("exact_gap.npy", allow_pickle=True).item()',
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
            status = (
                f"{beta.size}/{N_BETA}, "
                f"last beta={beta[-1]:.6f}"
            )
        print(f"  h={h:g}: {status}", flush=True)


if __name__ == "__main__":
    main()
