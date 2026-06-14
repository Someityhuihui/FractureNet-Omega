"""
符号回归 — 发现断裂力学映射
=============================
使用 PySR 从生成的数据集中自动发现断裂力学公式。

目标:
  发现 σ_N(ft, β, p) 的解析表达式
  即: 名义应力 = f(抗拉强度, 脆性数, 牵引阶次)

预期输出示例:
  σ_N = 0.5 * ft / sqrt(1 + 2.5 * β) * (1 + 0.1 * (p-1))

使用说明:
  python discover_formula.py --input fracture_data.csv
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse

# ======================================================================
# 方法1: PySR 符号回归 (推荐)
# ======================================================================
def discover_with_pysr(df, feature_cols=None, target_col='sigma_N',
                       niterations=100, output_dir='output'):
    """
    使用 PySR 进行符号回归
    
    参数
    ----------
    df : pd.DataFrame
        输入数据集
    feature_cols : list, optional
        特征列名列表
    target_col : str
        目标列名
    niterations : int
        PySR 迭代次数
    output_dir : str
        输出目录
    
    返回
    ----------
    model : PySRRegressor
        训练好的符号回归模型
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("[!] PySR 未安装。请运行: pip install pysr")
        print("    也可以使用 discover_with_bruteforce() 进行简单搜索。")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 默认特征: 无量纲参数
    if feature_cols is None:
        feature_cols = ['ft', 'beta', 'p']

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"特征: {feature_cols}")
    print(f"样本数: {len(df)}")
    print(f"PySR 迭代次数: {niterations}")
    print()

    # 创建符号回归模型
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "exp", "log", "abs"],
        # 限制表达式复杂度
        maxsize=30,
        # 等式选择标准
        model_selection="best",
        # 保存结果
        procs=4,
        populations=8,
        verbosity=1,
        output_directory=output_dir,
    )

    # 训练
    print("开始训练...")
    model.fit(X, y)

    # 输出结果
    print("\n" + "=" * 60)
    print("发现的公式 (PySR)")
    print("=" * 60)
    print(model)

    # 保存公式到文本文件
    with open(os.path.join(output_dir, 'discovered_formula.txt'), 'w') as f:
        f.write(str(model))

    return model


# ======================================================================
# 方法2: 暴力搜索候选公式 (无需 PySR, 纯 NumPy)
# ======================================================================
def discover_with_bruteforce(df, feature_cols=None, target_col='sigma_N'):
    """
    使用预定义的候选公式模板进行暴力搜索
    
    适用于 PySR 无法安装的情况。
    通过拟合系数并比较 R² 来评估每个候选公式。

    参数
    ----------
    df : pd.DataFrame
        输入数据集
    feature_cols : list, optional
        特征列名列表
    target_col : str
        目标列名
    """
    if feature_cols is None:
        feature_cols = ['ft', 'beta', 'p']

    ft = df['ft'].values
    beta = df['beta'].values
    p = df['p'].values
    y_true = df[target_col].values

    # Additional features if available
    S_soft = df['S_soft'].values if 'S_soft' in df.columns else np.ones_like(ft)
    scode = df['softening_code'].values if 'softening_code' in df.columns else np.zeros_like(ft)

    # 候选公式模板 (用 c0, c1, ... 表示待定系数)
    candidates = [
        # Bazant size effect law variants
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta),
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta) * (1 + c[2] * (p - 1)),
        lambda ft, beta, p, c: c[0] * ft / (1 + c[1] * beta)**c[2],
        lambda ft, beta, p, c: c[0] * ft / (1 + c[1] * beta**c[2]),
        # p-dependent variants
        lambda ft, beta, p, c: c[0] * ft / (np.sqrt(1 + c[1] * beta) + c[2] * (p - 1)),
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta) * (p)**c[2],
        # Exponential forms
        lambda ft, beta, p, c: c[0] * ft * np.exp(-c[1] * beta),
        lambda ft, beta, p, c: c[0] * ft * np.exp(-c[1] * beta) * (1 + c[2] * (p - 1)),
        # Polynomial forms
        lambda ft, beta, p, c: c[0] * ft * (1 - c[1] * beta + c[2] * beta**2),
        lambda ft, beta, p, c: c[0] * ft / (1 + c[1] * beta + c[2] * p),
        # EXTENDED: With softening shape factor S_soft
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta / S_soft) * (1 + c[2] * (p - 1)),
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta * S_soft) * (1 + c[2] * (p - 1)),
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta) * S_soft**c[2],
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta * (2*p+1)/3 / S_soft),
        # EXTENDED: With softening code
        lambda ft, beta, p, c: c[0] * ft / np.sqrt(1 + c[1] * beta) * (1 + c[2] * (p - 1) + c[3] * scode),
    ]

    names = [
        "sigma = c0*ft/sqrt(1 + c1*beta)",
        "sigma = c0*ft/sqrt(1 + c1*beta) * (1 + c2*(p-1))",
        "sigma = c0*ft/(1 + c1*beta)^c2",
        "sigma = c0*ft/(1 + c1*beta^c2)",
        "sigma = c0*ft/(sqrt(1 + c1*beta) + c2*(p-1))",
        "sigma = c0*ft/sqrt(1 + c1*beta) * p^c2",
        "sigma = c0*ft*exp(-c1*beta)",
        "sigma = c0*ft*exp(-c1*beta)*(1 + c2*(p-1))",
        "sigma = c0*ft*(1 - c1*beta + c2*beta^2)",
        "sigma = c0*ft/(1 + c1*beta + c2*p)",
        "sigma = c0*ft/sqrt(1 + c1*beta/S_soft) * (1 + c2*(p-1))",
        "sigma = c0*ft/sqrt(1 + c1*beta*S_soft) * (1 + c2*(p-1))",
        "sigma = c0*ft/sqrt(1 + c1*beta) * S_soft^c2",
        "sigma = c0*ft/sqrt(1 + c1*beta*(2p+1)/3/S_soft)",
        "sigma = c0*ft/sqrt(1 + c1*beta)*(1 + c2*(p-1) + c3*s_code)",
    ]

    print("暴力搜索候选公式...")
    print(f"样本数: {len(df)}")
    print()

    best_r2 = -np.inf
    best_idx = -1
    best_coeffs = None

    from scipy.optimize import curve_fit

    for i, (func, name) in enumerate(zip(candidates, names)):
        try:
            # Determine number of free coefficients
            n_coeffs = len(func(ft, beta, p, [1, 1, 1, 1]))
            default_p0 = [0.5, 2.5, 0.1, 0.05]
            p0 = default_p0[:n_coeffs]

            # Curve fitting (S_soft, scode captured as closures)
            popt, _ = curve_fit(
                lambda X, *c: func(X[0], X[1], X[2], list(c)),
                (ft, beta, p),
                y_true,
                p0=p0,
                maxfev=20000
            )

            # 预测和 R²
            y_pred = func(ft, beta, p, popt)
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - ss_res / ss_tot

            # RMSE
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))

            print(f"[{i+1:2d}] R²={r2:.6f}  RMSE={rmse:.4f}  "
                  f"系数={np.array2string(popt, precision=4, separator=', ')}")
            print(f"      公式: {name}")

            if r2 > best_r2:
                best_r2 = r2
                best_idx = i
                best_coeffs = popt

        except Exception as e:
            print(f"[{i+1:2d}] 失败: {name} — {str(e)[:60]}")

    print()
    print("=" * 60)
    print(f"最优候选 [#{best_idx+1}]:")
    print(f"  公式: {names[best_idx]}")
    print(f"  系数: {np.array2string(best_coeffs, precision=6, separator=', ')}")
    print(f"  R²  = {best_r2:.6f}")
    print("=" * 60)

    return best_idx, best_coeffs, names, candidates


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='μPF-CZM 符号回归发现断裂力学映射')
    parser.add_argument('--input', type=str, default='fracture_data.csv',
                        help='输入数据文件 (CSV)')
    parser.add_argument('--method', type=str, choices=['pysr', 'bruteforce', 'both'],
                        default='both', help='符号回归方法')
    parser.add_argument('--iterations', type=int, default=100,
                        help='PySR 迭代次数 (仅 pysr 方法)')
    parser.add_argument('--output', type=str, default='output',
                        help='输出目录')
    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.input)
    print(f"加载数据: {args.input} ({len(df)} 行)")

    # 执行符号回归
    if args.method in ('pysr', 'both'):
        print("\n" + "=" * 60)
        print("方法1: PySR 符号回归")
        print("=" * 60)
        model = discover_with_pysr(
            df,
            feature_cols=['ft', 'beta', 'p'],
            target_col='sigma_N',
            niterations=args.iterations,
            output_dir=args.output
        )

    if args.method in ('bruteforce', 'both'):
        print("\n" + "=" * 60)
        print("方法2: 暴力搜索候选公式 (纯 NumPy/ SciPy)")
        print("=" * 60)
        discover_with_bruteforce(
            df,
            feature_cols=['ft', 'beta', 'p'],
            target_col='sigma_N'
        )
