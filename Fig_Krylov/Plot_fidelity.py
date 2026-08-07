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
# Load data
# ============================================================

N_BETA = 20

krylov_results = np.load(
    "data_fidelity_and_gap.npy",
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
    Load beta and Krylov absolute gaps.

    Returns
    -------
    beta : ndarray, shape (n,)
    gap : ndarray, shape (n, len(m_values))
    """
    entry = get_entry(
        krylov_results,
        h,
        "data_fidelity_and_gap.npy",
    )

    beta = np.asarray(
        entry["beta"],
        dtype=float,
    )

    gap = np.asarray(
        entry["fidelity"],
        dtype=float,
    )

    if beta.ndim != 1:
        raise ValueError(
            f"U={h:g}: Krylov beta must be one-dimensional; "
            f"received shape {beta.shape}."
        )

    expected_shape = (
        beta.size,
        len(m_values),
    )

    if gap.shape != expected_shape:
        raise ValueError(
            f"U={h:g}: Krylov absolute_gap has shape {gap.shape}, "
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

    if not np.all(np.isfinite(gap)):
        raise ValueError(
            f"U={h:g}: Krylov absolute_gap contains NaN or inf."
        )

    if np.any(gap <= 0.0):
        raise ValueError(
            f"U={h:g}: Krylov absolute_gap contains nonpositive values."
        )

    return beta, gap


krylov_datasets = [
    load_krylov_h(h)
    for h in h_values
]

# ============================================================
# Print truncation information
# ============================================================

for h, krylov_data in zip(
    h_values,
    krylov_datasets,
):
    krylov_beta = krylov_data[0]

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


    print(
        f"U={h:g}: "
        f"Krylov [{krylov_status}], "
    )


# ============================================================
# LaTeX linear tick formatter
# ============================================================

def latex_linear_formatter(value, position):
    """Format a linear tick using bold LaTeX."""
    if np.isclose(value, 0.0):
        value = 0.0

    return rf"$\bm{{{value:g}}}$"


x_formatter = ticker.FuncFormatter(
    latex_linear_formatter
)

y_formatter = ticker.FuncFormatter(
    latex_linear_formatter
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
    (krylov_beta, krylov_gap),
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
                krylov_gap[:, column],
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

    ax.set_xlim(
        0.0,
        8.0,
    )

    ax.set_ylim(
        0.0,
        1.1,
    )

    # x major ticks: 0, 2, 4, 6, 8
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(2)
    )

    # x minor ticks: spacing 1
    ax.xaxis.set_minor_locator(
        ticker.MultipleLocator(1)
    )

    # Dynamic linear y ticks. This avoids clipping when gaps exceed 1.
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
            min_n_ticks=4,
        )
    )

    ax.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(2)
    )

    ax.xaxis.set_major_formatter(
        x_formatter
    )

    ax.yaxis.set_major_formatter(
        y_formatter
    )

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
        r"\textbf{Ground state fidelity}"
    )


# ============================================================
# Layout and save
# ============================================================

fig.tight_layout()

fig.savefig(
    "krylov_fidelity.pdf",
    bbox_inches="tight",
)

plt.show()