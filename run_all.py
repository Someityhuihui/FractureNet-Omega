"""
FractureNet-Ω 主入口
======================
统一运行整个流水线: 数据生成 → 符号回归 → 验证可视化

用法:
  python run_all.py                   # 运行完整流水线
  python run_all.py --skip-pysr       # 跳过 PySR (使用暴力搜索)
  python run_all.py --stage generate  # 只运行数据生成
  python run_all.py --stage discover  # 只运行符号回归
  python run_all.py --stage validate  # 只运行可视化验证
"""

import argparse
import subprocess
import sys
import os


def run_generate():
    """阶段2: 批量数据生成"""
    print("\n" + "=" * 60)
    print("阶段2: 批量数据生成")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, 'generate_data.py', '--output', 'fracture_data.csv'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def run_discover(method='both', niterations=100):
    """阶段3: 符号回归"""
    print("\n" + "=" * 60)
    print("阶段3: 符号回归发现映射")
    print("=" * 60)
    cmd = [
        sys.executable, 'discover_formula.py',
        '--input', 'fracture_data.csv',
        '--method', method,
        '--iterations', str(niterations),
        '--output', 'output'
    ]
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def run_validate():
    """阶段4: 验证与可视化"""
    print("\n" + "=" * 60)
    print("阶段4: 验证与可视化")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, 'visualize.py', '--data', 'fracture_data.csv',
         '--output', 'figures'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def check_setup():
    """检查环境配置"""
    print("=" * 60)
    print("FractureNet-Ω 环境检查")
    print("=" * 60)

    checks = []

    # Python 版本
    py_ver = sys.version
    checks.append(("Python", f"{py_ver.split()[0]}", True))

    # NumPy
    try:
        import numpy as np
        checks.append(("NumPy", np.__version__, True))
    except ImportError:
        checks.append(("NumPy", "未安装", False))

    # SciPy
    try:
        import scipy as sp
        checks.append(("SciPy", sp.__version__, True))
    except ImportError:
        checks.append(("SciPy", "未安装", False))

    # Matplotlib
    try:
        import matplotlib as mpl
        checks.append(("Matplotlib", mpl.__version__, True))
    except ImportError:
        checks.append(("Matplotlib", "未安装", False))

    # Pandas
    try:
        import pandas as pd
        checks.append(("Pandas", pd.__version__, True))
    except ImportError:
        checks.append(("Pandas", "未安装", False))

    # PySR (可选)
    try:
        import pysr
        checks.append(("PySR", pysr.__version__, True))
    except ImportError:
        checks.append(("PySR", "未安装 (可选)", False))

    # 打印检查结果
    all_ok = True
    for name, ver, ok in checks:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  [{status}] {name:<15} {ver}")
        if not ok and name != "PySR":
            all_ok = False

    print()
    if all_ok:
        print("环境检查通过。准备就绪!")
    else:
        print("请安装缺失的依赖: pip install -r requirements.txt")
    print()

    return all_ok


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='FractureNet-Ω: μPF-CZM 断裂力学符号回归发现框架'
    )
    parser.add_argument('--stage', type=str,
                        choices=['all', 'generate', 'discover', 'validate', 'check'],
                        default='all', help='运行阶段')
    parser.add_argument('--method', type=str,
                        choices=['pysr', 'bruteforce', 'both'],
                        default='both', help='符号回归方法')
    parser.add_argument('--iterations', type=int, default=100,
                        help='PySR 迭代次数')
    parser.add_argument('--skip-pysr', action='store_true',
                        help='跳过 PySR (使用暴力搜索)')
    args = parser.parse_args()

    # 处理 --skip-pysr
    if args.skip_pysr and args.method == 'both':
        args.method = 'bruteforce'

    # 运行指定阶段
    if args.stage == 'check':
        check_setup()
        sys.exit(0)

    # 先检查环境
    if not check_setup():
        sys.exit(1)

    success = True

    if args.stage in ('all', 'generate'):
        if run_generate() != 0:
            print("[!] 数据生成失败")
            success = False

    if success and args.stage in ('all', 'discover'):
        if run_discover(method=args.method, niterations=args.iterations) != 0:
            print("[!] 符号回归失败")
            success = False

    if success and args.stage in ('all', 'validate'):
        if run_validate() != 0:
            print("[!] 验证可视化失败")
            success = False

    print()
    if success:
        print("=" * 60)
        print("FractureNet-Ω 流水线运行完成!")
        print("=" * 60)
        print("  数据集:    fracture_data.csv")
        print("  发现公式:  output/discovered_formula.txt")
        print("  可视化图:  figures/")
        print("=" * 60)
    else:
        print("[!] 流水线运行未完全成功，请检查上述错误。")
        sys.exit(1)
