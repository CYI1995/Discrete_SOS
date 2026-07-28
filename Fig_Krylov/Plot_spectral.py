import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Plot settings
# ============================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'text.usetex': True,
    'text.latex.preamble':
        r'\usepackage{times}\usepackage{amsmath,amssymb}',
    'axes.unicode_minus': False,
})

# ============================================================
# Load data
# ============================================================
L = 20
Listofbeta = np.linspace(1/L,8,L) 

Listofnorm1 = np.load('Listofspectralgap_h0.01.npy')
Listofnorm2 = np.load('Listofspectralgap_h0.1.npy')
Listofnorm3 = np.load('Listofspectralgap_h1.0.npy')
Listofnorm4 = np.load('Listofspectralgap_h10.0.npy')

datasets = [
    Listofnorm1,
    Listofnorm2,
    Listofnorm3,
    Listofnorm4
]

h_values = [0.01, 0.1, 1, 10]

# ============================================================
# Data convention:
#
# column 0 : exact parent Hamiltonian
# column 1 : Krylov m = 12
# column 2 : Krylov m = 24
# column 3 : Krylov m = 36
# ============================================================
m_values = [12, 24, 36]

colors = ['royalblue', 'red', 'black']
markers = ['^', 'o', 's']

# ============================================================
# Create figure
# ============================================================
fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    sharex=True,
    sharey=True
)

for ax, norm, h in zip(axes.flat, datasets, h_values):

    # --------------------------------------------------------
    # Exact parent Hamiltonian
    # --------------------------------------------------------
    ax.plot(
        Listofbeta,
        norm[:, 0],
        label=r'$\boldsymbol{\mathrm{Exact}}$',
        color='darkgreen',
        linestyle='-',
        linewidth=2.5
    )

    # --------------------------------------------------------
    # Krylov approximations
    # --------------------------------------------------------
    for j, (m, c, mk) in enumerate(
        zip(m_values, colors, markers)
    ):
        ax.plot(
            Listofbeta,
            norm[:, j + 1],   # Important: columns 1,2,3
            label=rf'$\boldsymbol{{m = {m}}}$',
            color=c,
            marker=mk,
            linestyle='--',
            linewidth=1.8,
            markersize=5
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(-0.1,10.1)
    ax.set_title(
        rf'$\boldsymbol{{h = {h}}}$'
    )

    ax.legend()

# ============================================================
# Axis labels
# ============================================================
for ax in axes[1, :]:
    ax.set_xlabel(
        r'$\boldsymbol{\beta}$'
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        r'$\boldsymbol{\mathrm{Spectral\ gap}}$'
    )

# ============================================================
# Layout
# ============================================================
fig.tight_layout()

fig.savefig(
    'KrylovSpectralGap.pdf',
    bbox_inches='tight'
)

plt.show()