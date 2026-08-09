import numpy as np
import scipy
import math
from scipy import linalg
# import source as mycode
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 开启真 LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''

'''

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'text.usetex': True,
    'text.latex.preamble':
        r'\usepackage{times}'
        r'\usepackage{amsmath,amssymb}',
    'axes.unicode_minus': False,
})


# ============================================================
# Load data
# ============================================================
L = 20
Listofbeta = np.linspace(1 / L, 8, L)

Listofnorm_h001 = np.load('Listofnorm_h0.01.npy')
Listofnorm_h01 = np.load('Listofnorm_h0.1.npy')
Listofnorm_h1 = np.load('Listofnorm_h1.0.npy')
Listofnorm_h10 = np.load('Listofnorm_h10.0.npy')

# 注意数据与 h 的顺序保持一致
datasets = [
    Listofnorm_h001,
    Listofnorm_h01,
    Listofnorm_h1,
    Listofnorm_h10,
]

h_values = [0.01, 0.1, 1, 10]

colors = ['royalblue', 'red', 'black']
markers = ['^', 'o', 's']
m_values = [8, 12, 16]


# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True,
    sharey=True
)

for ax, norm, h in zip(axes.flat, datasets, h_values):

    for col, (m, color, marker) in enumerate(
        zip(m_values, colors, markers)
    ):
        ax.plot(
            Listofbeta,
            norm[:, col],
            label=rf'$\boldsymbol{{m={m}}}$',
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
        )

    ax.set_yscale('log')
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-14, 1)

    ax.set_title(
        rf'$\boldsymbol{{U={h:g}}}$'
    )

    # Put the common legend only in the fourth subplot
    axes[1, 1].legend(
        loc='best',
        frameon=True,
    )

    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=6,
        width=1.2,
    )

    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        length=3,
        width=1.0,
    )


# ============================================================
# 横坐标刻度：与参考代码相同，手动设置 LaTeX 粗体
# ============================================================
x_ticks = np.array([0, 2, 4, 6, 8])

for ax in axes.flat:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([
        rf'$\boldsymbol{{{int(x)}}}$'
        for x in x_ticks
    ])


# ============================================================
# 纵坐标刻度：手动设置为粗体 10^n
# ============================================================
y_exponents = np.array([-14, -12, -10, -8, -6, -4, -2, 0])
y_ticks = 10.0 ** y_exponents

for ax in axes.flat:
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([
        rf'$\boldsymbol{{10^{{{exponent}}}}}$'
        for exponent in y_exponents
    ])


# ============================================================
# Axis labels
# ============================================================
for ax in axes[1, :]:
    ax.set_xlabel(
        r'$\boldsymbol{\beta}$'
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        r'$\boldsymbol{'
        r'\varepsilon_{\max}'
        r'}$'
    )


fig.tight_layout()

fig.savefig(
    'krylov_max_error.pdf',
    bbox_inches='tight',
    pad_inches=0.02,
    dpi=600,
)

plt.show()