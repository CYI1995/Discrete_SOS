"""
Ground-state infidelity 1 - F of the Krylov parent Hamiltonian.

Reads data_fidelity_and_gap.npy (written by the computation script) and
produces krylov_infidelity.pdf: a 2x2 panel, one per h, with one curve per
Krylov order m.

Infidelity source
-----------------
If the data file provides an "infidelity" field, it is used directly.  That
field is the numerically stable one: it should be computed as

    1 - |<u, v>|^2 = || v - <u, v> u ||^2

for normalized u, v, which retains full relative precision.

Otherwise 1 - fidelity is formed from the stored fidelity.  That subtraction
cancels catastrophically as F -> 1: a stored double equal to 1.0 carries no
information about how far below one the true value was, and anything below
about 1e-16 in the resulting plot is quantization of the fidelity rather than
a physical result.  A warning is printed when this path is taken.

Axes
----
The y range and its ticks are fixed by Y_EXPONENTS below, in the style of the
epsilon_max figure.  Because the range is hard-coded, points outside it are
silently clipped by matplotlib; the loader counts them and reports the count,
so a badly chosen range is caught rather than quietly hidden.

Zeros cannot be drawn on a log axis, so exactly-zero infidelities are masked
rather than floored -- a floor would draw a flat line at an arbitrary height
and read as a genuine plateau.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


# ============================================================
# Global LaTeX configuration
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",

    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath,amssymb}"
        r"\usepackage{bm}"
        r"\usepackage{newtxtext,newtxmath}"
    ),

    "axes.unicode_minus": False,
})


# ============================================================
# Parameters
# ============================================================

N_BETA = 20

DATA_PATH = "data_fidelity_and_gap.npy"
FIGURE_PATH = "krylov_infidelity.pdf"

# Below this the difference 1 - F is dominated by double-precision
# quantization of F.  Used only for the warning, never to clip data.
PRECISION_FLOOR = 1e-16

# Fixed y ticks and range, as in the epsilon_max figure.  Edit this one line
# to move the window; the limits and the labels both follow from it.
Y_EXPONENTS = np.arange(-14, -5, 2)          # -14, -12, ..., 0
Y_LIMITS = (10.0 ** float(Y_EXPONENTS[0]), 10.0 ** float(Y_EXPONENTS[-1]))

X_TICKS = np.array([0, 2, 4, 6, 8])
X_LIMITS = (0.0, 8.0)


# ============================================================
# Load data
# ============================================================

krylov_results = np.load(
    DATA_PATH,
    allow_pickle=True,
).item()

h_values = [0.01, 0.1, 1.0, 10.0]


# Prefer the metadata stored with the Krylov data.
if (
    "metadata" in krylov_results
    and "m_list" in krylov_results["metadata"]
):
    m_values = np.asarray(
        krylov_results["metadata"]["m_list"],
        dtype=int,
    )
else:
    # Use this only if the old file has no metadata.
    m_values = np.array(
        [8, 16, 24],
        dtype=int,
    )

if len(m_values) != 3:
    raise ValueError(
        "The plotting style currently defines three colors and markers, "
        f"but m_values contains {len(m_values)} entries."
    )

colors = [
    "royalblue",
    "red",
    "black",
]

markers = [
    "^",
    "o",
    "s",
]


# ============================================================
# Data validation helpers
# ============================================================

def numerical_keys(dictionary):
    """Return only numerical h/U keys, excluding metadata."""
    return sorted(
        key
        for key in dictionary
        if isinstance(
            key,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    )


def get_entry(dictionary, h, filename):
    """Get the entry corresponding to h."""
    if h in dictionary:
        return dictionary[h]

    # Robust fallback for floating-point dictionary keys.
    for key in numerical_keys(dictionary):
        if np.isclose(float(key), float(h)):
            return dictionary[key]

    raise KeyError(
        f"{filename} does not contain U={h:g}. "
        f"Available numerical keys: {numerical_keys(dictionary)}"
    )


def load_krylov_h(h):
    """
    Load beta and the Krylov ground-state infidelity.

    Returns
    -------
    beta : ndarray, shape (n,)
    infidelity : masked ndarray, shape (n, len(m_values))
        Exact zeros are masked so they are dropped by the log axis.
    derived : bool
        True when the infidelity was obtained as 1 - fidelity, and is
        therefore limited by cancellation near F = 1.
    """
    entry = get_entry(
        krylov_results,
        h,
        DATA_PATH,
    )

    beta = np.asarray(
        entry["beta"],
        dtype=float,
    )

    if "infidelity" in entry:
        infidelity = np.asarray(
            entry["infidelity"],
            dtype=float,
        )
        derived = False
    else:
        fidelity = np.asarray(
            entry["fidelity"],
            dtype=float,
        )

        if fidelity.size and (
            np.any(fidelity < -1e-12)
            or np.any(fidelity > 1.0 + 1e-12)
        ):
            raise ValueError(
                f"U={h:g}: Krylov fidelity lies outside [0, 1]."
            )

        infidelity = 1.0 - fidelity
        derived = True

    if beta.ndim != 1:
        raise ValueError(
            f"U={h:g}: Krylov beta must be one-dimensional; "
            f"received shape {beta.shape}."
        )

    expected_shape = (
        beta.size,
        len(m_values),
    )

    if infidelity.shape != expected_shape:
        raise ValueError(
            f"U={h:g}: Krylov infidelity has shape {infidelity.shape}, "
            f"but the expected shape is {expected_shape}. "
            "Check that m_values agrees with the calculation."
        )

    if beta.size > 1 and not np.all(
        np.diff(beta) > 0.0
    ):
        raise ValueError(
            f"U={h:g}: Krylov beta values must be strictly increasing."
        )

    if not np.all(np.isfinite(beta)):
        raise ValueError(
            f"U={h:g}: Krylov beta contains NaN or inf."
        )

    if not np.all(np.isfinite(infidelity)):
        raise ValueError(
            f"U={h:g}: Krylov infidelity contains NaN or inf."
        )

    if infidelity.size and np.any(infidelity < -1e-12):
        raise ValueError(
            f"U={h:g}: Krylov infidelity contains negative values."
        )

    # F = 1 to working precision gives 1 - F = 0, which a log axis cannot
    # place.  Masking drops those markers; flooring them would invent a
    # plateau that looks like a result.
    infidelity = np.ma.masked_less_equal(infidelity, 0.0)

    return beta, infidelity, derived


krylov_datasets = []
derived_flags = []

for h in h_values:
    beta_h, infidelity_h, derived_h = load_krylov_h(h)
    krylov_datasets.append((beta_h, infidelity_h))
    derived_flags.append(derived_h)

infidelity_is_derived = any(derived_flags)

if infidelity_is_derived:
    print(
        "warning: no 'infidelity' field in the data file; using 1 - F. "
        f"Values below ~{PRECISION_FLOOR:.0e} are cancellation noise, "
        "not physics.",
        flush=True,
    )


# ============================================================
# Print truncation and clipping information
# ============================================================

for h, krylov_data in zip(
    h_values,
    krylov_datasets,
):
    krylov_beta, krylov_infidelity = krylov_data

    if krylov_beta.size == 0:
        krylov_status = "no valid beta"
    elif krylov_beta.size < N_BETA:
        krylov_status = (
            f"{krylov_beta.size}/{N_BETA}, "
            f"last beta={krylov_beta[-1]:.4f}"
        )
    else:
        krylov_status = (
            f"{krylov_beta.size}/{N_BETA}, complete"
        )

    masked_count = int(
        np.ma.getmaskarray(krylov_infidelity).sum()
    )

    if masked_count:
        krylov_status += (
            f", {masked_count} point(s) with 1-F = 0 dropped"
        )

    # The y range is hard-coded, so anything outside it vanishes without
    # comment.  Count those points here instead.
    visible_values = np.asarray(
        np.ma.compressed(krylov_infidelity),
        dtype=float,
    )

    below = int(np.sum(visible_values < Y_LIMITS[0]))
    above = int(np.sum(visible_values > Y_LIMITS[1]))

    if below or above:
        krylov_status += (
            f", {below} below / {above} above the plotted y range"
        )

    print(
        f"U={h:g}: "
        f"Krylov [{krylov_status}], "
    )


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

for (
    ax,
    (krylov_beta, krylov_infidelity),
    h,
) in zip(
    axes.flat,
    krylov_datasets,
    h_values,
):

    # Krylov curves
    if krylov_beta.size > 0:
        for column, (m, color, marker) in enumerate(
            zip(
                m_values,
                colors,
                markers,
            )
        ):
            ax.plot(
                krylov_beta,
                krylov_infidelity[:, column],
                label=rf"$\bm{{m={m}}}$",
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
            )

    if (
        krylov_beta.size == 0
    ):
        ax.text(
            0.5,
            0.5,
            r"\textbf{no resolvable} $\bm{\beta}$",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_yscale("log")

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)

    ax.set_title(
        rf"$\bm{{U={h:g}}}$"
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.2,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3,
        width=1.0,
    )


# ============================================================
# x ticks: bold LaTeX integers, set by hand
# ============================================================

for ax in axes.flat:
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([
        rf"$\bm{{{int(x)}}}$"
        for x in X_TICKS
    ])

    # x minor ticks: spacing 1
    ax.xaxis.set_minor_locator(
        ticker.MultipleLocator(1)
    )


# ============================================================
# y ticks: bold LaTeX 10^n, set by hand
# ============================================================

y_ticks = 10.0 ** Y_EXPONENTS.astype(float)

for ax in axes.flat:
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([
        rf"$\bm{{10^{{{int(exponent)}}}}}$"
        for exponent in Y_EXPONENTS
    ])

    # set_yticks fixes the majors only; keep unlabelled sub-decade minors.
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(
            base=10.0,
            subs=tuple(np.arange(2, 10) * 0.1),
        )
    )

    ax.yaxis.set_minor_formatter(
        ticker.NullFormatter()
    )


# ============================================================
# Common legend in the fourth subplot
# ============================================================

legend_axis = axes[1, 1]

handles, labels = (
    legend_axis.get_legend_handles_labels()
)

# Fallback if the fourth panel has no data.
if not handles:
    for source_axis in axes.flat:
        handles, labels = (
            source_axis.get_legend_handles_labels()
        )

        if handles:
            break

if handles:
    legend_axis.legend(
        handles,
        labels,
        loc="best",
        frameon=True,
    )


# ============================================================
# Axis labels
# ============================================================

for ax in axes[1, :]:
    ax.set_xlabel(
        r"$\bm{\beta}$"
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        r"$\textbf{Ground state infidelity}$"
    )


# ============================================================
# Layout and save
# ============================================================

fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=600,
)

plt.show()