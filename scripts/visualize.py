"""
验证与可视化脚本
===================
加载生成的数据和发现的公式，进行可视化验证。

功能:
  1. 加载并展示数据集摘要
  2. 绘制参数空间分布
  3. 验证发现的公式 (预测 vs 真实)
  4. 绘制牵引-分离曲线族
  5. 绘制尺寸效应曲线
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os
import sys
import argparse

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from micro_pf_czm_1d import MicroPF_CZM_1D


# ======================================================================
# 图1: 数据摘要 — 参数分布直方图
# ======================================================================
def plot_data_summary(df, output_dir='figures'):
    """绘制数据集中各参数的分布直方图"""
    os.makedirs(output_dir, exist_ok=True)

    cols = ['E', 'ft', 'Gf', 'p', 'b', 'sigma_N']
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        ax = axes[i]
        if col in df.columns:
            if df[col].dtype == 'object':
                # 分类变量 (如 softening) 画柱状图
                counts = df[col].value_counts()
                ax.bar(counts.index, counts.values)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_data_summary.png'), dpi=150)
    print(f"  保存: {output_dir}/01_data_summary.png")
    plt.close()


# ======================================================================
# 图2: 预测 vs 真实 (公式验证)
# ======================================================================
def plot_prediction_vs_actual(df, formula_func=None, output_dir='figures'):
    """
    绘制预测 vs 真实散点图

    参数
    ----------
    df : pd.DataFrame
        数据集
    formula_func : callable, optional
        公式函数 f(ft, beta, p) -> sigma_N
        若不提供，则使用加载的数据作为"真实值"基准
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true = df['sigma_N'].values

    if formula_func is not None:
        ft = df['ft'].values
        beta = df['beta'].values
        p = df['p'].values
        y_pred = formula_func(ft, beta, p)
        title_base = '预测 vs 真实 (公式)'
        suffix = '02_prediction_vs_actual_formula'
    else:
        # 仅显示真实数据的分布
        y_pred = None
        title_base = '名义应力分布'
        suffix = '02_sigma_distribution'

    fig, axes = plt.subplots(1, 2 if y_pred is not None else 1,
                             figsize=(14, 5))

    if y_pred is not None:
        ax1, ax2 = axes

        # 散点图
        ax1.scatter(y_true, y_pred, alpha=0.3, s=10, c='blue')
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, 'r--', linewidth=2, label='完美预测')
        ax1.set_xlabel('真实 σ_N (MPa)')
        ax1.set_ylabel('预测 σ_N (MPa)')
        ax1.set_title(title_base)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 相对误差直方图
        error = (y_pred - y_true) / y_true * 100
        ax2.hist(error, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('相对误差 (%)')
        ax2.set_ylabel('频次')
        ax2.set_title(f'相对误差 (均值={np.mean(error):.2f}%, '
                      f'标准差={np.std(error):.2f}%)')
        ax2.grid(True, alpha=0.3)

    else:
        ax = axes if not hasattr(axes, '__len__') else axes[0]
        ax.hist(y_true, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('σ_N (MPa)')
        ax.set_ylabel('频次')
        ax.set_title('名义应力分布')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{suffix}.png'), dpi=150)
    print(f"  保存: {output_dir}/{suffix}.png")
    plt.close()


# ======================================================================
# 图3: 牵引-分离曲线族
# ======================================================================
def plot_traction_separation_family(output_dir='figures'):
    """绘制不同参数下的牵引-分离曲线族"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 不同 ft
    ax = axes[0, 0]
    for ft in [2.0, 3.0, 4.0, 5.0]:
        model = MicroPF_CZM_1D(E=30000, ft=ft, Gf=0.12, b=2.0)
        w, sigma = model.traction_separation_law()
        ax.plot(w, sigma, label=f'ft={ft} MPa')
    ax.set_xlabel('裂纹张开位移 w (mm)')
    ax.set_ylabel('牵引力 σ (MPa)')
    ax.set_title('不同抗拉强度 ft')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) 不同 Gf
    ax = axes[0, 1]
    for Gf in [0.08, 0.12, 0.16, 0.20]:
        model = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=Gf, b=2.0)
        w, sigma = model.traction_separation_law()
        ax.plot(w, sigma, label=f'Gf={Gf} N/mm')
    ax.set_xlabel('裂纹张开位移 w (mm)')
    ax.set_ylabel('牵引力 σ (MPa)')
    ax.set_title('不同断裂能 Gf')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) 不同 p
    ax = axes[1, 0]
    for p in [1.0, 1.5, 2.0]:
        model = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, p=p, b=2.0)
        w, sigma = model.traction_separation_law()
        ax.plot(w, sigma, label=f'p={p}')
    ax.set_xlabel('裂纹张开位移 w (mm)')
    ax.set_ylabel('牵引力 σ (MPa)')
    ax.set_title('不同牵引阶次 p')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) 不同 b
    ax = axes[1, 1]
    for b in [1.0, 2.0, 4.0]:
        model = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, b=b)
        w, sigma = model.traction_separation_law()
        ax.plot(w, sigma, label=f'b={b} mm')
    ax.set_xlabel('裂纹张开位移 w (mm)')
    ax.set_ylabel('牵引力 σ (MPa)')
    ax.set_title('不同长度尺度 b')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('牵引-分离曲线族 (μPF-CZM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_traction_separation_family.png'), dpi=150)
    print(f"  保存: {output_dir}/03_traction_separation_family.png")
    plt.close()


# ======================================================================
# 图4: 尺寸效应曲线
# ======================================================================
def plot_size_effect(output_dir='figures'):
    """绘制不同 p 下的尺寸效应曲线"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    L_chars = np.logspace(0, 3, 50)  # 1mm ~ 1000mm

    for p in [1.0, 1.5, 2.0]:
        sigma_N = []
        for Lc in L_chars:
            model = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, p=p, b=2.0)
            Pmax = model.peak_load(L_char=Lc)
            sigma_N.append(Pmax / (100 * 100))
        ax.loglog(L_chars, sigma_N, label=f'p={p}', linewidth=2)

    # Bazant 尺寸效应律渐近线
    ax.axhline(y=0.5*3.0, color='gray', linestyle='--', label='强度准则 (小结构)')
    # LEFM 渐近线 (示意)
    ax.axline((10, 2), (100, 0.63), color='gray', linestyle=':', label='LEFM (大结构)')

    ax.set_xlabel('特征尺寸 L_char (mm)')
    ax.set_ylabel('名义应力 σ_N (MPa)')
    ax.set_title('尺寸效应曲线 (μPF-CZM)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_size_effect.png'), dpi=150)
    print(f"  保存: {output_dir}/04_size_effect.png")
    plt.close()


# ======================================================================
# 图5: 参数敏感度分析
# ======================================================================
def plot_parameter_sensitivity(df, output_dir='figures'):
    """绘制各参数对名义应力的影响"""
    os.makedirs(output_dir, exist_ok=True)

    # 分类: 按 softening 分组分别绘制
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    params = ['E', 'ft', 'Gf', 'p', 'b', 'beta']
    for i, param in enumerate(params):
        ax = axes[i]
        if param not in df.columns:
            continue

        # 按 softening 着色
        for sf in df['softening'].unique():
            mask = df['softening'] == sf
            ax.scatter(df.loc[mask, param], df.loc[mask, 'sigma_N'],
                       label=sf, alpha=0.3, s=5)
        ax.set_xlabel(param)
        ax.set_ylabel('σ_N (MPa)')
        ax.set_title(f'{param} 对名义应力的影响')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_parameter_sensitivity.png'), dpi=150)
    print(f"  保存: {output_dir}/05_parameter_sensitivity.png")
    plt.close()


# ======================================================================
# 完整验证流水线
# ======================================================================
def run_full_validation(data_file='fracture_data.csv', output_dir='figures'):
    """
    运行完整验证流水线

    参数
    ----------
    data_file : str
        数据集 CSV 文件路径
    output_dir : str
        图像输出目录
    """
    print("=" * 50)
    print("μPF-CZM 验证与可视化")
    print("=" * 50)

    # 1. 加载数据
    df = pd.read_csv(data_file)
    print(f"\n[1/5] 加载数据: {data_file} ({len(df)} 行)")

    # 2. 数据摘要图
    print("[2/5] 绘制数据摘要...")
    plot_data_summary(df, output_dir)

    # 3. 名义应力分布
    print("[3/5] 绘制预测 vs 真实...")
    plot_prediction_vs_actual(df, formula_func=None, output_dir=output_dir)

    # 4. 牵引-分离曲线族
    print("[4/5] 绘制牵引-分离曲线族...")
    plot_traction_separation_family(output_dir)

    # 5. 尺寸效应和敏感度
    print("[5/5] 绘制尺寸效应和参数敏感度...")
    plot_size_effect(output_dir)
    plot_parameter_sensitivity(df, output_dir)

    print(f"\n所有图像已保存至 {output_dir}/ 目录")


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='μPF-CZM 验证与可视化')
    parser.add_argument('--data', type=str, default='fracture_data.csv',
                        help='输入数据文件')
    parser.add_argument('--output', type=str, default='figures',
                        help='图像输出目录')
    args = parser.parse_args()

    run_full_validation(data_file=args.data, output_dir=args.output)
    plt.show()
