"""
Diverse Fracture Dataset Generator
====================================
Generates physically accurate + diverse displacement/strain fields
using our uPF-CZM 1D/2D solvers + augmentation.

Output format per sample:
  {
    'displacement': (2, H, W) or (1, N) — u_x, u_y fields
    'phase_field': (H, W) or (N,)   — damage field d(x)
    'strain':       (H, W) or (N,)   — derived strain field
    'params':       dict             — E, ft, Gf, p, b, softening, geometry
    'label':        float            — peak load sigma_N / ft
  }

Strategy:
  50%  from 1D analytical solver (fast, diverse params)
  30%  from 2D FEM SENB (geometry + crack path realism)
  20%  from augmentation (rotation, noise, scaling)
"""

import numpy as np
import os, sys, json, time, argparse
from itertools import product
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from micro_pf_czm_1d import MicroPF_CZM_1D

# Try 2D import (may fail if scikit-fem not available)
try:
    from fem_2d import MuPFCZMMaterial, MuPFCZM2DSolver, find_senb_dofs
    from mesh_utils import generate_senb_mesh_tri, set_notch_initial_damage
    HAS_2D = True
except Exception:
    HAS_2D = False
    print("[WARN] 2D FEM not available — using 1D only")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ================================================================
# 1D Dataset Generator
# ================================================================
def generate_1d_samples(n_samples=500, seed=42):
    """
    Generate 1D fracture profiles with diverse parameters.

    Returns list of dicts with:
      - d_profile: phase-field d(x) along bar
      - w_profile: COD w(x) along bar
      - F_u_curve: (F, u) parametric curve
      - params: material + model parameters
      - sigma_N_ft: normalized peak strength
    """
    np.random.seed(seed)
    samples = []

    # Parameter ranges (broad coverage)
    E_range = (2000, 50000)
    ft_range = (1.5, 8.0)
    Gf_range = (0.02, 0.50)
    p_range = (1.0, 3.0)
    b_range = (1.0, 20.0)
    L_range = (20, 1000)
    soft_types = ['linear', 'exponential', 'cornelissen']

    for i in range(n_samples):
        # Random parameters
        E = 10 ** np.random.uniform(np.log10(E_range[0]), np.log10(E_range[1]))
        ft = np.random.uniform(*ft_range)
        Gf = np.random.uniform(*Gf_range)
        p = np.random.uniform(*p_range)
        b = np.random.uniform(*b_range)
        L = np.random.uniform(*L_range)
        softening = soft_types[i % len(soft_types)]

        try:
            m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening=softening, p=p, b=b)

            # Phase-field profile
            d_star = np.random.uniform(0.3, 0.95)
            x_d, d_vals = m.compute_d_profile(d_star, n_points=256)

            # COD profile
            x_w, w_vals = m.compute_cod_profile(d_star, n_points=256)

            # F-u curve
            d_sequence = np.linspace(0.01, 0.98, 50)
            F_vals = ft * (1 - d_sequence)**(2*p)
            u_vals = F_vals * 2*L/E + np.array([m.compute_center_cod(d) for d in d_sequence])

            # Peak load prediction
            detail = m.peak_load_detail(L_char=L)
            sigma_N_ft = detail['sigma_N'] / ft

            samples.append({
                'd_profile': d_vals,
                'w_profile': w_vals,
                'x_coord': x_d,
                'F_curve': F_vals,
                'u_curve': u_vals,
                'params': {
                    'E': float(E), 'ft': float(ft), 'Gf': float(Gf),
                    'softening': softening, 'p': float(p), 'b': float(b),
                    'L': float(L), 'lch': float(m.lch), 'd_star': float(d_star),
                },
                'label': float(sigma_N_ft),
                'type': '1d',
            })
        except Exception as e:
            pass

    print(f"  1D: {len(samples)} samples generated")
    return samples


# ================================================================
# 2D Dataset Generator (SENB)
# ================================================================
def generate_2d_samples(n_samples=50, seed=42):
    """
    Generate 2D SENB crack patterns with diverse geometry.

    Small mesh for speed, diverse params for variety.
    """
    if not HAS_2D:
        print("  2D: SKIPPED (no FEM available)")
        return []

    np.random.seed(seed)
    samples = []

    D_range = (12, 40)
    a0_ratio_range = (0.15, 0.35)
    E_range = (3000, 15000)
    ft_range = (2.0, 5.0)
    Gf_range = (0.06, 0.20)
    p_range = (1.0, 2.5)
    b_range = (2.0, 6.0)

    for i in range(n_samples):
        D = np.random.uniform(*D_range)
        a0 = np.random.uniform(*a0_ratio_range)
        E = np.random.uniform(*E_range)
        ft = np.random.uniform(*ft_range)
        Gf = np.random.uniform(*Gf_range)
        p = np.random.uniform(*p_range)
        b = np.random.uniform(*b_range)
        h_el = b / 4  # resolve length scale
        soft = ['linear', 'exponential', 'cornelissen'][i % 3]

        try:
            mesh, nf, bp = generate_senb_mesh_tri(D=D, a0_ratio=a0, h_el=h_el)
            mat = MuPFCZMMaterial(E=E, nu=0.2, Gf=Gf, ft=ft, softening=soft, p=p, b=b)
            solver = MuPFCZM2DSolver(mat, mesh)

            di = np.zeros(solver.ndof_d)
            set_notch_initial_damage(mesh, nf, di, b=b, smooth=True)
            solver.set_initial_damage(di)

            bc = find_senb_dofs(mesh, bp)
            supp = bc['supp_ux'] + bc['supp_uy'] + bc['supp_uy_r']
            load = bc['load_uy']

            # Run 3 steps
            u_max = 0.005 * D
            u_hist, d_hist = [], []
            for step in range(3):
                ub = -(step+1) * u_max / 3
                un, dn, _, _ = solver.solve_step(ub, load, supp, n_stagger=4, tol=1e-3)
                u_hist.append(un); d_hist.append(dn)

            # Extract displacement at crack zone
            cx = bp['center_x']
            crack_nodes = np.where(np.abs(mesh.p[0] - cx) < D*0.1)[0]
            if len(crack_nodes) > 0:
                u_y_crack = np.array([u_hist[-1][2*n+1] for n in crack_nodes])
                d_crack = np.array([d_hist[-1][n] for n in crack_nodes])
            else:
                u_y_crack = np.zeros(10); d_crack = np.zeros(10)

            samples.append({
                'mesh_nodes': mesh.p.T.tolist(),
                'mesh_tets': mesh.t.T.tolist(),
                'd_final': d_hist[-1].tolist(),
                'u_final': u_hist[-1].tolist(),
                'd_crack_zone': d_crack.tolist(),
                'u_crack_zone': u_y_crack.tolist(),
                'params': {
                    'D': float(D), 'a0_ratio': float(a0),
                    'E': float(E), 'ft': float(ft), 'Gf': float(Gf),
                    'softening': soft, 'p': float(p), 'b': float(b),
                    'lch': float(mat.lch),
                },
                'label': float(d_crack.max()) if len(d_crack) > 0 else 0.0,
                'type': '2d',
            })
        except Exception as e:
            pass

    print(f"  2D: {len(samples)} samples generated")
    return samples


# ================================================================
# Augmentation
# ================================================================
def augment_1d_samples(samples, n_aug_per_sample=10, seed=42):
    """Augment 1D profiles with noise, scaling, offset."""
    np.random.seed(seed)
    augmented = list(samples)

    for s in samples:
        for _ in range(n_aug_per_sample):
            noise_level = np.random.uniform(0.001, 0.03)
            scale = np.random.uniform(0.8, 1.2)
            offset = np.random.uniform(-0.02, 0.02)

            d_aug = s['d_profile'] * scale + offset + \
                np.random.randn(len(s['d_profile'])) * noise_level
            w_aug = s['w_profile'] * scale + offset + \
                np.random.randn(len(s['w_profile'])) * noise_level

            augmented.append({
                **{k: v for k, v in s.items() if k not in ['d_profile', 'w_profile']},
                'd_profile': d_aug,
                'w_profile': w_aug,
                'type': '1d_augmented',
                'aug_from': s['params'],
            })

    return augmented


# ================================================================
# Main Pipeline
# ================================================================
def generate_full_dataset(n_1d=500, n_2d=30, n_aug=8, output_dir=None):
    """
    Generate complete diverse dataset.

    Args:
        n_1d:  Number of 1D analytical base samples
        n_2d:  Number of 2D FEM samples
        n_aug: Augmentations per 1D sample
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    run_dir = os.path.join(output_dir, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 55)
    print(f"  Dataset Generation Pipeline")
    print(f"  Output: {run_dir}")
    print("=" * 55)

    # Phase 1: 1D analytical (fast, many)
    print("\n[1/3] 1D Analytical Profiles...")
    t0 = time.time()
    samples_1d = generate_1d_samples(n_1d)
    print(f"       Time: {time.time()-t0:.1f}s")

    # Phase 2: 2D FEM (slower, fewer)
    print(f"\n[2/3] 2D FEM SENB Patterns...")
    t0 = time.time()
    samples_2d = generate_2d_samples(n_2d)
    print(f"       Time: {time.time()-t0:.1f}s")

    # Phase 3: Augmentation
    print(f"\n[3/3] Augmentation ({n_aug}x per 1D sample)...")
    t0 = time.time()
    all_1d = augment_1d_samples(samples_1d, n_aug)
    total = len(all_1d) + len(samples_2d)
    print(f"       Total samples: {total} ({len(all_1d)} 1D + {len(samples_2d)} 2D)")
    print(f"       Time: {time.time()-t0:.1f}s")

    # Save
    print(f"\nSaving...")
    save_1d = os.path.join(run_dir, '1d_samples.npz')
    save_2d = os.path.join(run_dir, '2d_samples.npz')

    # 1D: extract arrays
    n_1d_total = len(all_1d)
    d_profiles = np.array([s['d_profile'] for s in all_1d])
    w_profiles = np.array([s['w_profile'] for s in all_1d])
    labels_1d = np.array([s['label'] for s in all_1d])
    params_1d = [s['params'] for s in all_1d]

    np.savez_compressed(save_1d,
                        d_profiles=d_profiles, w_profiles=w_profiles,
                        labels=labels_1d)
    with open(os.path.join(run_dir, '1d_params.json'), 'w') as f:
        json.dump(params_1d, f, indent=1)

    # 2D
    if samples_2d:
        d_2d = [np.array(s['d_crack_zone']) for s in samples_2d]
        labels_2d = np.array([s['label'] for s in samples_2d])
        params_2d = [s['params'] for s in samples_2d]
        np.savez_compressed(save_2d, labels=labels_2d)
        with open(os.path.join(run_dir, '2d_params.json'), 'w') as f:
            json.dump(params_2d, f, indent=1)

    # Manifest
    manifest = {
        'timestamp_utc': ts,
        'n_1d_base': n_1d,
        'n_1d_augmented': len(all_1d),
        'n_2d': len(samples_2d),
        'total': total,
        'aug_factor': n_aug,
        'd_profile_shape': list(d_profiles.shape),
        'files': {
            '1d': '1d_samples.npz',
            '2d': '2d_samples.npz' if samples_2d else None,
            '1d_params': '1d_params.json',
            '2d_params': '2d_params.json' if samples_2d else None,
        },
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    size_mb = sum(os.path.getsize(os.path.join(run_dir, fn))
                  for fn in os.listdir(run_dir)) / 1e6
    print(f"\n{'='*55}")
    print(f"  DATASET COMPLETE")
    print(f"  {total} samples, {size_mb:.1f} MB")
    print(f"  {run_dir}")
    print(f"{'='*55}")
    return run_dir, manifest


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fracture dataset')
    parser.add_argument('--n_1d', type=int, default=500)
    parser.add_argument('--n_2d', type=int, default=30)
    parser.add_argument('--aug', type=int, default=8)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.n_1d = 200
        args.n_2d = 0   # skip 2D (scipy slow)
        args.aug = 5

    generate_full_dataset(args.n_1d, args.n_2d, args.aug, args.output)
