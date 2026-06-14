"""
批量数据生成脚本
=================
对参数空间扫描，运行 μPF-CZM 求解器，生成断裂力学数据集。

输出: CSV 文件，包含每组参数下的:
  - 输入参数: E, ft, Gf, softening, p, b
  - 派生参数: lch, a0, eta0
  - 输出: Pmax (峰值荷载), D (裂纹带半宽度)
"""

import numpy as np
import pandas as pd
from itertools import product
import os
import sys
import time

# 将 src 目录加入路径，导入求解器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from micro_pf_czm_1d import MicroPF_CZM_1D


# ======================================================================
# 参数空间定义
# ======================================================================
PARAM_SPACE = {
    'E': [20000, 30000, 40000],            # 杨氏模量 (MPa)
    'ft': [2.0, 3.0, 4.0, 5.0],           # 抗拉强度 (MPa)
    'Gf': [0.08, 0.12, 0.16, 0.20],       # 断裂能 (N/mm)
    'softening': ['linear', 'exponential', 'cornelissen'],
    'p': [1.0, 1.5, 2.0],                 # 牵引阶次
    'b': [1.0, 2.0],                      # 相场长度尺度 (mm)
}

# 结构参数 (峰值荷载计算用)
STRUCT_PARAMS = {
    'L_char': 100.0,   # 特征尺寸 (mm)
    'width': 100.0,    # 截面宽度 (mm)
    'height': 100.0,   # 截面高度 (mm)
    'B_factor': 0.5,   # 几何因子
}


# ======================================================================
# 单次仿真函数
# ======================================================================
def run_simulation(param_dict, struct_params=None):
    """
    对一组参数运行仿真，提取关键输出

    参数
    ----------
    param_dict : dict
        包含 E, ft, Gf, softening, p, b
    struct_params : dict, optional
        包含 L_char, width, height, B_factor

    返回
    ----------
    results : dict
        输入参数 + 派生参数 + 输出量
    """
    if struct_params is None:
        struct_params = STRUCT_PARAMS

    # 创建求解器
    model = MicroPF_CZM_1D(
        E=param_dict['E'],
        ft=param_dict['ft'],
        Gf=param_dict['Gf'],
        softening=param_dict['softening'],
        p=param_dict['p'],
        b=param_dict['b']
    )

    # EXTENDED peak load with full diagnostics
    detail = model.peak_load_detail(
        L_char=struct_params['L_char'],
        width=struct_params['width'],
        height=struct_params['height'],
        B_factor=struct_params['B_factor']
    )

    # 裂纹带半宽度 (在 d*=0.8 处评估)
    D = model.crack_half_width(d_star=0.8)

    # Effective fracture energy and process zone
    Gf_eff = model.effective_fracture_energy_xi()
    l_pz = model.process_zone_length()

    # 收集结果
    results = {
        # --- 输入参数 ---
        'E': param_dict['E'],
        'ft': param_dict['ft'],
        'Gf': param_dict['Gf'],
        'softening': param_dict['softening'],
        'p': param_dict['p'],
        'b': param_dict['b'],
        # --- 派生参数 ---
        'lch': model.lch,
        'a0': model.a0,
        'eta0': model.eta0,
        'sigma0': model.sigma0,
        'beta': detail['beta'],
        # --- 扩展诊断量 ---
        'R_p': detail['R_p'],
        'S_soft': detail['S_soft'],
        'f_b': detail['f_b'],
        'beta_eff': detail['beta_eff'],
        # --- 输出量 ---
        'Pmax': detail['P_max'],
        'sigma_N': detail['sigma_N'],
        'crack_half_width': D,
        'Gf_eff': Gf_eff,
        'process_zone': l_pz,
    }

    return results


# ======================================================================
# 批量运行
# ======================================================================
def batch_generate(output_file='fracture_data.csv', verbose=True):
    """
    对参数空间所有组合批量运行仿真

    参数
    ----------
    output_file : str
        输出 CSV 文件路径
    verbose : bool
        是否打印进度

    返回
    ----------
    df : pd.DataFrame
        完整数据集
    """
    keys = list(PARAM_SPACE.keys())
    values = list(PARAM_SPACE.values())
    combinations = list(product(*values))

    total = len(combinations)
    if verbose:
        print(f"总参数组合数: {total}")
        print(f"  维度: {dict((k, len(v)) for k, v in PARAM_SPACE.items())}")
        print()

    all_results = []
    t_start = time.time()

    for idx, combo in enumerate(combinations, 1):
        param_dict = dict(zip(keys, combo))

        try:
            results = run_simulation(param_dict)
            all_results.append(results)

            if verbose and idx % 50 == 0:
                elapsed = time.time() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"  进度: {idx}/{total} ({100*idx/total:.1f}%) | "
                      f"速率: {rate:.1f} 组/秒")

        except Exception as e:
            if verbose:
                print(f"  [!] 失败: E={param_dict['E']}, ft={param_dict['ft']}, "
                      f"Gf={param_dict['Gf']}, softening={param_dict['softening']}, "
                      f"p={param_dict['p']}, b={param_dict['b']}")
                print(f"      错误: {e}")

    # 构建 DataFrame
    df = pd.DataFrame(all_results)

    # 添加无量纲组合特征（便于后续符号回归）
    df['ft_over_E'] = df['ft'] / df['E']
    df['Gf_over_E'] = df['Gf'] / df['E']
    df['lch_over_b'] = df['lch'] / df['b']
    df['sigma_N_over_ft'] = df['sigma_N'] / df['ft']
    df['b_over_lch'] = df['b'] / df['lch']

    # 软化类型数值编码
    soft_map = {'linear': 0, 'exponential': 1, 'cornelissen': 2, 'ppr': 3}
    df['softening_code'] = df['softening'].map(soft_map)

    # 保存
    df.to_csv(output_file, index=False)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n完成! 共 {len(df)} 条记录 (失败 {total - len(df)} 组)")
        print(f"耗时: {elapsed:.1f} 秒")
        print(f"输出文件: {output_file}")
        print(f"\n数据列: {list(df.columns)}")

    return df


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='μPF-CZM 批量数据生成')
    parser.add_argument('--output', type=str, default='fracture_data.csv',
                        help='输出 CSV 文件路径')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，不输出进度')
    args = parser.parse_args()

    df = batch_generate(output_file=args.output, verbose=not args.quiet)

    # 显示数据摘要
    print("\n数据摘要:")
    print(df.describe())
