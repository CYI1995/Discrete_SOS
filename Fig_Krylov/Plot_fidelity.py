import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

# ============================================================
# 全局 LaTeX 配置
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
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
    'text.latex.preamble': (
        r'\usepackage{amsmath,amssymb}'
        r'\usepackage{bm}'
        r'\usepackage{newtxtext,newtxmath}'
    ),

    'axes.unicode_minus': False,
})

# ============================================================
# 读取数据
# ============================================================
L = 20
Listofbeta = np.linspace(1 / L, 8, L)

Listofnorm_h001 = np.load('Listoffidelity_h0.01.npy')
Listofnorm_h01 = np.load('Listoffidelity_h0.1.npy')
Listofnorm_h1 = np.load('Listoffidelity_h1.0.npy')
Listofnorm_h10 = np.load('Listoffidelity_h10.0.npy')

# 数据和标题严格保持相同顺序
datasets = [
    Listofnorm_h001,
    Listofnorm_h01,
    Listofnorm_h1,
    Listofnorm_h10,
]

h_values = [0.01, 0.1, 1, 10]

colors = ['royalblue', 'red', 'black']
markers = ['^', 'o', 's']
m_values = [12, 24, 36]

# ============================================================
# LaTeX 线性刻度格式化器
# ============================================================
def latex_linear_formatter(x, pos):
    """
    线性坐标格式，例如：
    0, 0.2, 0.4, ..., 1
    """
    if np.isclose(x, 0):
        x = 0

    return rf'$\bm{{{x:g}}}$'


x_formatter = ticker.FuncFormatter(latex_linear_formatter)
y_formatter = ticker.FuncFormatter(latex_linear_formatter)

# ============================================================
# 作图
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

for ax, norm, h in zip(axes.flat, datasets, h_values):

    for col, (m, color, marker) in enumerate(
        zip(m_values, colors, markers)
    ):
        ax.plot(
            Listofbeta,
            norm[:, col],
            label=rf'$\bm{{m={m}}}$',
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
        )

    # ========================================================
    # 线性坐标轴范围
    # ========================================================
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.1, 1.1)

    # x 轴主刻度：0, 2, 4, 6, 8
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(2)
    )

    # x 轴次刻度：间隔 1
    ax.xaxis.set_minor_locator(
        ticker.MultipleLocator(1)
    )

    # y 轴主刻度：0, 0.2, 0.4, ..., 1
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(0.2)
    )

    # y 轴次刻度：间隔 0.1
    ax.yaxis.set_minor_locator(
        ticker.MultipleLocator(0.1)
    )

    # 横纵坐标统一使用 LaTeX 粗体格式
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)

    # 标题与图例
    ax.set_title(
        rf'$\bm{{U={h:g}}}$'
    )
    ax.legend(
        frameon=True
    )

    # 主刻度线
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=6,
        width=1.2,
    )

    # 次刻度线
    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        length=3,
        width=1.0,
    )

# ============================================================
# 坐标轴标签
# ============================================================
for ax in axes[1, :]:
    ax.set_xlabel(
        r'$\bm{\beta}$'
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        r'\textbf{Ground state fidelity}'
    )

# ============================================================
# 调整布局并保存
# ============================================================
fig.tight_layout()

fig.savefig(
    'GroundStateFidelity.pdf',
    bbox_inches='tight'
)

plt.show()