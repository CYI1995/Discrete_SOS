#!/usr/bin/env python3
"""
Two-panel figure formatted after the supplied reference style:
vertical stack, Times New Roman, bold labels/titles/tick labels,
explicit tick placement, and PDF/PNG output at 600 dpi.

Top (a):    cost versus beta.
Bottom (b): trace-distance upper bound versus normalized time.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# ============================================================
# Parameters
# ============================================================

DATA_DIR = Path(".")

COST_FILE = "data_cost.npy"
MIXING_FILE = "data_mixing.npy"

OUTPUT_PDF = "cost_and_mixing_vertical.pdf"
OUTPUT_PNG = "cost_and_mixing_vertical.png"

FIGSIZE = (8.5, 12)
DPI = 600


# ------------------------------------------------------------
# Panel (a): x-axis
# ------------------------------------------------------------
# Use None for automatic 1/2/2.5/5 × 10^k spacing.
COST_XLIM = None
COST_XTICK_STEP = None
COST_XTICK_TARGET = 6


# ------------------------------------------------------------
# Panel (b): x-axis
# ------------------------------------------------------------
MIXING_XLIM = None
MIXING_XTICK_STEP = None
MIXING_XTICK_TARGET = 6


# ------------------------------------------------------------
# Logarithmic y-axes
# ------------------------------------------------------------
# Set to (lo, hi) to manually specify the displayed range.
# Both values must be positive.
COST_YLIM = None
MIXING_YLIM = None

# Padding added to an automatically fitted logarithmic range.
LOG_PAD_DECADES = 0.08


# ============================================================
# Global Matplotlib settings
# ============================================================

def _usetex_available():
    """Check whether a sufficiently complete LaTeX installation exists."""
    if shutil.which("latex") is None:
        return False

    if shutil.which("dvipng") is None:
        return False

    try:
        result = subprocess.run(
            ["kpsewhich", "type1ec.sty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(result.stdout.strip())

    except Exception:
        return False


HAVE_LATEX = _usetex_available()

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "text.usetex": HAVE_LATEX,
    "axes.unicode_minus": False,
})

if HAVE_LATEX:
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{times}"
        r"\usepackage{amsmath,amssymb}"
    )


# ============================================================
# Loading and validation
# ============================================================

def load_dictionary(filename):
    """Load a dictionary saved using np.save(..., dictionary)."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find required data file: {path.resolve()}"
        )

    loaded = np.load(path, allow_pickle=True)

    try:
        data = loaded.item()
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"{filename} must contain a dictionary saved by np.save."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{filename} does not contain a dictionary."
        )

    return data


def load_array(data, key, dataset_name):
    """Load and validate a one-dimensional array from a dictionary."""
    if key not in data:
        raise KeyError(
            f"{dataset_name} does not contain key {key!r}."
        )

    array = np.asarray(data[key], dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"{dataset_name}[{key!r}] must be one-dimensional; "
            f"received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{dataset_name}[{key!r}] contains NaN or infinite values."
        )

    return array


def load_first_available(data, keys, dataset_name):
    """Load the first key present in a list of possible keys."""
    for key in keys:
        if key in data:
            return load_array(data, key, dataset_name)

    raise KeyError(
        f"{dataset_name} contains none of the expected keys: {keys}."
    )


def validate_lengths(dataset_name, **arrays):
    """Check that all arrays in a dataset have equal length."""
    lengths = {
        name: len(array)
        for name, array in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{dataset_name} arrays have inconsistent lengths: "
            f"{lengths}."
        )


# Load dictionaries.
cost_data = load_dictionary(COST_FILE)
mixing_data = load_dictionary(MIXING_FILE)


# Load cost data.
beta_values = load_array(
    cost_data,
    "beta",
    "cost data",
)

cost_gaussian = load_array(
    cost_data,
    "cost_SGauss",
    "cost data",
)

cost_identity = load_array(
    cost_data,
    "cost_SI",
    "cost data",
)

cost_optimal = load_array(
    cost_data,
    "cost_Sopt",
    "cost data",
)

validate_lengths(
    "cost data",
    beta=beta_values,
    gaussian=cost_gaussian,
    identity=cost_identity,
    optimal=cost_optimal,
)


# Load mixing data.
time_values = load_array(
    mixing_data,
    "time",
    "mixing data",
)

mixing_gaussian = load_first_available(
    mixing_data,
    (
        "mixing_SGauss",
        "mixing_bound_SGauss",
    ),
    "mixing data",
)

mixing_identity = load_first_available(
    mixing_data,
    (
        "mixing_SI",
        "mixing_bound_SI",
    ),
    "mixing data",
)

mixing_optimal = load_first_available(
    mixing_data,
    (
        "mixing_Sopt",
        "mixing_bound_Sopt",
    ),
    "mixing data",
)

validate_lengths(
    "mixing data",
    time=time_values,
    gaussian=mixing_gaussian,
    identity=mixing_identity,
    optimal=mixing_optimal,
)


# Sort both datasets by their horizontal coordinates.
cost_order = np.argsort(beta_values)

beta_values = beta_values[cost_order]
cost_gaussian = cost_gaussian[cost_order]
cost_identity = cost_identity[cost_order]
cost_optimal = cost_optimal[cost_order]


mixing_order = np.argsort(time_values)

time_values = time_values[mixing_order]
mixing_gaussian = mixing_gaussian[mixing_order]
mixing_identity = mixing_identity[mixing_order]
mixing_optimal = mixing_optimal[mixing_order]


# All values must be positive because both vertical axes are logarithmic.
for name, values in {
    "cost_SGauss": cost_gaussian,
    "cost_SI": cost_identity,
    "cost_Sopt": cost_optimal,
    "mixing_SGauss": mixing_gaussian,
    "mixing_SI": mixing_identity,
    "mixing_Sopt": mixing_optimal,
}.items():

    if np.any(values <= 0):
        raise ValueError(
            f"{name} contains nonpositive values and cannot be "
            "shown on a logarithmic y-axis."
        )


# ============================================================
# Explicit bold tick helpers
# ============================================================

def nice_step(span, target=6):
    """
    Return a tick spacing of the form

        1, 2, 2.5, 5, or 10 × 10^k

    producing approximately `target` intervals.
    """
    if span <= 0:
        return 1.0

    raw = span / target
    magnitude = 10.0 ** np.floor(np.log10(raw))

    for multiplier in (
        1.0,
        2.0,
        2.5,
        5.0,
        10.0,
    ):
        if raw <= multiplier * magnitude:
            return multiplier * magnitude

    return 10.0 * magnitude


def make_ticks(lo, hi, step):
    """Generate ticks on a regular grid inside [lo, hi]."""
    first = np.ceil(
        lo / step - 1e-9
    ) * step

    ticks = np.arange(
        first,
        hi + 0.5 * step,
        step,
    )

    tolerance = 1e-9 * max(
        abs(lo),
        abs(hi),
        abs(step),
        1.0,
    )

    return ticks[
        (ticks >= lo - tolerance)
        & (ticks <= hi + tolerance)
    ]


def auto_decimals(values, max_decimals=8):
    """
    Determine the smallest number of decimal places for which
    all tick labels remain distinct.
    """
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0

    for decimals in range(max_decimals + 1):
        rounded = np.round(values, decimals)

        if len(np.unique(rounded)) == len(values):
            return decimals

    return max_decimals


def bold_dec_labels(values, decimals=3):
    """Create explicit bold LaTeX decimal labels."""
    labels = []

    if decimals > 0:
        zero_tolerance = 0.5 * 10.0 ** (-decimals)
    else:
        zero_tolerance = 0.5

    for value in values:
        if abs(value) < zero_tolerance:
            value = 0.0

        if decimals == 0:
            text = f"{value:.0f}"
        else:
            text = f"{value:.{decimals}f}"

        labels.append(
            rf"$\boldsymbol{{{text}}}$"
        )

    return labels


def bold_log_labels(exponents):
    """Create explicit bold labels of the form 10^k."""
    return [
        rf"$\boldsymbol{{10^{{{int(exponent)}}}}}$"
        for exponent in exponents
    ]


def configure_linear_xaxis(
    ax,
    values,
    forced_xlim,
    tick_step,
    target,
):
    """Configure a linear x-axis using explicit bold tick labels."""
    values = np.asarray(values, dtype=float)

    if forced_xlim is None:
        lo = float(np.min(values))
        hi = float(np.max(values))
    else:
        lo, hi = (
            float(value)
            for value in forced_xlim
        )

    if not hi > lo:
        raise ValueError(
            f"Empty x-axis range: {(lo, hi)}"
        )

    step = float(
        tick_step
        or nice_step(
            hi - lo,
            target=target,
        )
    )

    ticks = make_ticks(
        lo,
        hi,
        step,
    )

    if len(ticks) == 0:
        ticks = np.array(
            [lo, hi],
            dtype=float,
        )

    decimals = auto_decimals(ticks)

    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        bold_dec_labels(
            ticks,
            decimals,
        )
    )


def configure_log_yaxis(
    ax,
    curves,
    forced_ylim=None,
):
    """
    Configure a logarithmic y-axis with explicit bold
    major tick labels of the form 10^k.
    """
    all_values = np.concatenate([
        np.asarray(curve, dtype=float)
        for curve in curves
    ])

    if forced_ylim is None:
        data_lo = float(np.min(all_values))
        data_hi = float(np.max(all_values))

        log_lo = (
            np.log10(data_lo)
            - LOG_PAD_DECADES
        )

        log_hi = (
            np.log10(data_hi)
            + LOG_PAD_DECADES
        )

        y_lo = 10.0 ** log_lo
        y_hi = 10.0 ** log_hi

    else:
        y_lo, y_hi = (
            float(value)
            for value in forced_ylim
        )

    if y_lo <= 0 or not y_hi > y_lo:
        raise ValueError(
            f"Invalid logarithmic y-axis range: "
            f"{(y_lo, y_hi)}"
        )

    exponent_lo = int(
        np.floor(
            np.log10(y_lo)
        )
    )

    exponent_hi = int(
        np.ceil(
            np.log10(y_hi)
        )
    )

    exponents = np.arange(
        exponent_lo,
        exponent_hi + 1,
        dtype=int,
    )

    # Limit the number of labeled decades when the range is wide.
    stride = max(
        1,
        int(
            np.ceil(
                len(exponents) / 8
            )
        ),
    )

    exponents = exponents[::stride]

    if exponents[-1] != exponent_hi:
        exponents = np.append(
            exponents,
            exponent_hi,
        )

    major_ticks = 10.0 ** exponents

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(
        bold_log_labels(exponents)
    )

    # Keep logarithmic minor ticks, but suppress their labels.
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
        )
    )

    ax.yaxis.set_minor_formatter(
        NullFormatter()
    )


def style_spines_and_ticks(ax):
    """Apply the same spine and tick thickness to both panels."""
    ax.tick_params(
        axis="both",
        which="major",
        width=1.4,
        length=6,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        width=1.0,
        length=3,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


# ============================================================
# Create vertical subplots
# ============================================================

fig, (
    ax_top,
    ax_bottom,
) = plt.subplots(
    2,
    1,
    figsize=FIGSIZE,
    sharex=False,
)


# ============================================================
# Panel (a): cost comparison
# ============================================================

ax_top.set_title(
    r"$\boldsymbol{\mathrm{(a)\ Cost\ comparison}}$"
)

ax_top.plot(
    beta_values,
    cost_gaussian,
    marker="^",
    markersize=7,
    linewidth=1.9,
    color="crimson",
    label=r"$\boldsymbol{S_{\mathrm{Gaussian}}}$",
)

ax_top.plot(
    beta_values,
    cost_identity,
    marker="o",
    markersize=7,
    linewidth=1.9,
    color="royalblue",
    label=r"$\boldsymbol{S_{\mathrm{id}}}$",
)

ax_top.plot(
    beta_values,
    cost_optimal,
    marker="s",
    markersize=7,
    linewidth=1.9,
    color="forestgreen",
    label=r"$\boldsymbol{S_{\mathrm{opt}}}$",
)

ax_top.set_xlabel(
    r"$\boldsymbol{\beta}$"
)

ax_top.set_ylabel(
    r"$\boldsymbol{\mathrm{cost}(\beta)}$"
)

configure_linear_xaxis(
    ax_top,
    beta_values,
    COST_XLIM,
    COST_XTICK_STEP,
    COST_XTICK_TARGET,
)

configure_log_yaxis(
    ax_top,
    (
        cost_gaussian,
        cost_identity,
        cost_optimal,
    ),
    COST_YLIM,
)

style_spines_and_ticks(ax_top)

ax_top.legend(
    loc="upper left"
)


# ============================================================
# Panel (b): mixing comparison
# ============================================================

ax_bottom.set_title(
    r"$\boldsymbol{\mathrm{(b)\ Relaxation\ comparison}}$"
)

ax_bottom.plot(
    time_values,
    mixing_gaussian,
    marker="^",
    markersize=7,
    linewidth=1.9,
    color="crimson",
    label=r"$\boldsymbol{S_{\mathrm{Gaussian}}}$",
)

ax_bottom.plot(
    time_values,
    mixing_identity,
    marker="o",
    markersize=7,
    linewidth=1.9,
    color="royalblue",
    label=r"$\boldsymbol{S_{\mathrm{id}}}$",
)

ax_bottom.plot(
    time_values,
    mixing_optimal,
    marker="s",
    markersize=7,
    linewidth=1.9,
    color="forestgreen",
    label=r"$\boldsymbol{S_{\mathrm{opt}}}$",
)

ax_bottom.set_xlabel(
    r"$\boldsymbol{\|\mathrm{M}_{\mathcal{H}}\|t}$"
)

ax_bottom.set_ylabel(
    r"$\boldsymbol{\mathrm{Trace\ distance\ upper\ bound}}$"
)

configure_linear_xaxis(
    ax_bottom,
    time_values,
    MIXING_XLIM,
    MIXING_XTICK_STEP,
    MIXING_XTICK_TARGET,
)

configure_log_yaxis(
    ax_bottom,
    (
        mixing_gaussian,
        mixing_identity,
        mixing_optimal,
    ),
    MIXING_YLIM,
)

style_spines_and_ticks(ax_bottom)

ax_bottom.legend(
    loc="upper right"
)


# ============================================================
# Final layout and output
# ============================================================

plt.tight_layout(
    rect=[
        0.02,
        0.02,
        1.0,
        0.98,
    ],
    h_pad=2.5,
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=DPI,
)

plt.savefig(
    OUTPUT_PNG,
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=DPI,
)

print(
    f"saved: {OUTPUT_PDF} / {OUTPUT_PNG}"
)

plt.show()