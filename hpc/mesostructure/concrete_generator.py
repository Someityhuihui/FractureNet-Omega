"""
Concrete Mesostructure Generator
=================================
Generates realistic 2D/3D concrete microstructures with:
  - Random aggregate particles (graded by Fuller curve)
  - Cement mortar matrix
  - Interfacial Transition Zone (ITZ)
  - Voxel-based or geometry-based representation

Output: voxel arrays (2D/3D) + mesh-compatible material maps for FEM.

Aggregate gradation follows Fuller curve:
  P(d) = 100 * (d/d_max)^n, n = 0.45-0.70
  where P(d) is the cumulative % passing sieve size d.

Usage:
  python concrete_generator.py --dim 3 --size 100 --vol_frac 0.45
"""

import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass
import sys, os, argparse, json


@dataclass
class ConcreteSpec:
    """Concrete mix design specification."""
    Lx: float = 100.0     # domain size x (mm)
    Ly: float = 100.0     # domain size y (mm)
    Lz: float = 100.0     # domain size z (mm, only for 3D)
    voxel_size: float = 0.5   # voxel resolution (mm)
    vol_frac_agg: float = 0.45  # aggregate volume fraction
    d_min: float = 4.0     # min aggregate diameter (mm)
    d_max: float = 16.0    # max aggregate diameter (mm)
    n_sieves: int = 5      # number of sieve sizes
    fuller_n: float = 0.5  # Fuller curve exponent
    itz_thickness: float = 0.5  # ITZ thickness (mm)
    seed: int = 42         # random seed


def fuller_passing(d, d_max, n=0.5):
    """Fuller curve: cumulative % passing sieve size d."""
    return (d / d_max) ** n


def graded_diameters(d_min, d_max, n_sieves, fuller_n=0.5):
    """Generate graded aggregate diameters following Fuller curve."""
    # Generate sieve sizes
    d_sieves = np.geomspace(d_min, d_max, n_sieves + 1)
    cum_passing = fuller_passing(d_sieves, d_max, fuller_n)

    # Volume fraction retained on each sieve
    retained = -np.diff(cum_passing)
    retained = retained / retained.sum()

    return d_sieves, retained


def generate_2d_concrete(spec: ConcreteSpec):
    """
    Generate a 2D concrete mesostructure voxel array.

    Phases: 0=void, 1=mortar, 2=aggregate, 3=ITZ

    Returns
    -------
    voxels : ndarray (nx, ny) — phase labels
    agg_info : dict — aggregate positions and diameters
    """
    np.random.seed(spec.seed)

    nx = int(spec.Lx / spec.voxel_size)
    ny = int(spec.Ly / spec.voxel_size)

    # Initialize as mortar
    voxels = np.ones((nx, ny), dtype=np.int8)
    agg_info = {'positions': [], 'diameters': []}

    # Generate aggregate list
    d_sieves, vol_fracs = graded_diameters(
        spec.d_min, spec.d_max, spec.n_sieves, spec.fuller_n)

    # Total aggregate area needed
    A_total = spec.Lx * spec.Ly * spec.vol_frac_agg

    # Generate aggregates (take-and-place)
    for sieve_idx, (d_lo, d_hi) in enumerate(zip(d_sieves[:-1], d_sieves[1:])):
        d_avg = (d_lo + d_hi) / 2
        vol_frac = vol_fracs[sieve_idx]
        area_for_sieve = A_total * vol_frac
        area_per_agg = np.pi * (d_avg/2)**2
        n_expected = int(area_for_sieve / area_per_agg)

        placed = 0
        max_attempts = n_expected * 50
        for attempt in range(max_attempts):
            if placed >= n_expected:
                break

            # Random position (with margin)
            r = d_avg / 2
            cx = np.random.uniform(r, spec.Lx - r)
            cy = np.random.uniform(r, spec.Ly - r)

            # Check overlap with existing aggregates
            too_close = False
            for j in range(max(0, len(agg_info['positions'])-50), len(agg_info['positions'])):
                px, py = agg_info['positions'][j]
                pd = agg_info['diameters'][j]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < (r + pd/2) * 1.05:
                    too_close = True
                    break

            if not too_close:
                agg_info['positions'].append((cx, cy))
                agg_info['diameters'].append(d_avg)
                placed += 1

    # Paint aggregates onto voxel grid
    np.random.seed(spec.seed)
    d_sieves2, vol_fracs2 = graded_diameters(
        spec.d_min, spec.d_max, spec.n_sieves, spec.fuller_n)
    A_total2 = spec.Lx * spec.Ly * spec.vol_frac_agg

    painted_positions = []
    painted_diameters = []

    for s_idx, (d_lo, d_hi) in enumerate(zip(d_sieves2[:-1], d_sieves2[1:])):
        d_avg = (d_lo + d_hi) / 2
        vf = vol_fracs2[s_idx]
        area_for_sieve = A_total2 * vf
        area_per_agg = np.pi * (d_avg/2)**2
        n_exp = int(area_for_sieve / area_per_agg)
        placed = 0
        for _ in range(n_exp * 50):
            if placed >= n_exp: break
            r = d_avg/2
            cx = np.random.uniform(r, spec.Lx-r)
            cy = np.random.uniform(r, spec.Ly-r)
            too_close = False
            for j in range(max(0, len(painted_positions)-50), len(painted_positions)):
                px, py = painted_positions[j]
                pd = painted_diameters[j]
                dist = np.sqrt((cx-px)**2 + (cy-py)**2)
                if dist < (r + pd/2)*1.05:
                    too_close = True; break
            if not too_close:
                # Paint aggregate circle
                grid_x = np.arange(nx) * spec.voxel_size + spec.voxel_size/2
                grid_y = np.arange(ny) * spec.voxel_size + spec.voxel_size/2
                gx, gy = np.meshgrid(grid_x, grid_y, indexing='ij')
                dists = np.sqrt((gx - cx)**2 + (gy - cy)**2)

                # Aggregate core
                voxels[dists < r] = 2
                # ITZ ring
                itz_mask = (dists >= r) & (dists < r + spec.itz_thickness)
                voxels[itz_mask & (voxels == 1)] = 3

                painted_positions.append((cx, cy))
                painted_diameters.append(d_avg)
                placed += 1

    agg_info = {'positions': painted_positions, 'diameters': painted_diameters}
    return voxels, agg_info


def generate_3d_concrete(spec: ConcreteSpec):
    """
    Generate a 3D concrete mesostructure voxel array.

    Uses take-and-place with spherical aggregates.

    Returns
    -------
    voxels : ndarray (nx, ny, nz) — phase labels
    agg_info : dict
    """
    np.random.seed(spec.seed)
    nx = int(spec.Lx / spec.voxel_size)
    ny = int(spec.Ly / spec.voxel_size)
    nz = int(spec.Lz / spec.voxel_size)

    voxels = np.ones((nx, ny, nz), dtype=np.int8)
    d_sieves, vol_fracs = graded_diameters(
        spec.d_min, spec.d_max, spec.n_sieves, spec.fuller_n)
    V_total = spec.Lx * spec.Ly * spec.Lz * spec.vol_frac_agg

    print(f"  3D Concrete: {nx}x{ny}x{nz} voxels "
          f"({nx*ny*nz/1e6:.1f}M), vol_frac={spec.vol_frac_agg}")

    positions = []; diameters = []

    for s_idx, (d_lo, d_hi) in enumerate(zip(d_sieves[:-1], d_sieves[1:])):
        d_avg = (d_lo + d_hi) / 2
        vf = vol_fracs[s_idx]
        vol_for_sieve = V_total * vf
        vol_per_agg = 4/3 * np.pi * (d_avg/2)**3
        n_exp = max(1, int(vol_for_sieve / vol_per_agg))
        placed = 0

        for _ in range(min(n_exp * 100, 5000)):
            if placed >= n_exp: break
            r = d_avg / 2
            cx = np.random.uniform(r, spec.Lx-r)
            cy = np.random.uniform(r, spec.Ly-r)
            cz = np.random.uniform(r, spec.Lz-r)

            too_close = any(
                np.sqrt((cx-px)**2+(cy-py)**2+(cz-pz)**2) < (r+pd/2)*1.05
                for px, py, pz, pd in zip(
                    [p[0] for p in positions[-100:]],
                    [p[1] for p in positions[-100:]],
                    [p[2] for p in positions[-100:]],
                    diameters[-100:])
            ) if positions else False

            if not too_close:
                # Paint sphere (approximate — only near voxels)
                ci, cj, ck = int(cx/spec.voxel_size), int(cy/spec.voxel_size), int(cz/spec.voxel_size)
                r_vox = int(np.ceil((r + spec.itz_thickness) / spec.voxel_size))
                i0, i1 = max(0, ci-r_vox), min(nx, ci+r_vox+1)
                j0, j1 = max(0, cj-r_vox), min(ny, cj+r_vox+1)
                k0, k1 = max(0, ck-r_vox), min(nz, ck+r_vox+1)
                gi, gj, gk = np.mgrid[i0:i1, j0:j1, k0:k1]
                dists = np.sqrt(
                    ((gi+0.5)*spec.voxel_size - cx)**2 +
                    ((gj+0.5)*spec.voxel_size - cy)**2 +
                    ((gk+0.5)*spec.voxel_size - cz)**2
                )
                voxels[gi, gj, gk][dists < r] = 2
                itz_mask = (dists >= r) & (dists < r + spec.itz_thickness)
                voxels[gi, gj, gk][itz_mask & (voxels[gi,gj,gk] == 1)] = 3

                positions.append((cx, cy, cz))
                diameters.append(d_avg)
                placed += 1

    vol_agg = (voxels == 2).sum()
    vol_itz = (voxels == 3).sum()
    total_vol = voxels.size
    print(f"  Actual agg vol%: {100*vol_agg/total_vol:.1f}%, "
          f"ITZ vol%: {100*vol_itz/total_vol:.1f}%")
    return voxels, {'positions': positions, 'diameters': diameters}


def assign_material_properties(voxels, spec):
    """
    Assign mechanical properties to each phase.

    Returns
    -------
    E_field, nu_field, ft_field, Gf_field : ndarray
        Material property fields matching voxel dimensions.
    """
    # Typical concrete properties
    props = {
        1: {'E': 25000, 'nu': 0.2, 'ft': 3.0, 'Gf': 0.10},   # mortar
        2: {'E': 50000, 'nu': 0.18, 'ft': 6.0, 'Gf': 0.20},  # aggregate
        3: {'E': 15000, 'nu': 0.22, 'ft': 1.5, 'Gf': 0.03},  # ITZ (weakest)
    }

    shape = voxels.shape
    E = np.ones(shape) * 25000.0
    nu = np.ones(shape) * 0.2
    ft = np.ones(shape) * 3.0
    Gf = np.ones(shape) * 0.10

    for phase, p in props.items():
        mask = voxels == phase
        E[mask] = p['E']
        nu[mask] = p['nu']
        ft[mask] = p['ft']
        Gf[mask] = p['Gf']

    return E, nu, ft, Gf


def homogenize_effective_properties(voxels, spec, load_direction='y'):
    """
    Compute effective (homogenized) material properties using Voigt-Reuss bounds.

    Simple rule-of-mixtures for quick estimate.
    For rigorous homogenization, use computational homogenization with FEM.

    Returns dict with E_eff, nu_eff, ft_eff, Gf_eff.
    """
    E, nu, ft, Gf = assign_material_properties(voxels, spec)

    vol_fracs = {}
    for phase in [1, 2, 3]:
        vol_fracs[phase] = (voxels == phase).mean()

    # Voigt bound (upper): E_eff = sum(v_i * E_i)
    E_voigt = sum(vol_fracs.get(p, 0) * {
        1: 25000, 2: 50000, 3: 15000}[p] for p in [1, 2, 3])

    # Reuss bound (lower): 1/E_eff = sum(v_i / E_i)
    E_reuss = 1.0 / sum(vol_fracs.get(p, 0) / {
        1: 25000, 2: 50000, 3: 15000}[p] for p in [1, 2, 3])

    # Effective strength: weakest phase dominates (ITZ → ft_eff ≈ ft_ITZ)
    ft_eff = ft.mean()

    return {
        'E_voigt': E_voigt, 'E_reuss': E_reuss, 'E_eff': np.sqrt(E_voigt * E_reuss),
        'nu_eff': nu.mean(), 'ft_eff': ft_eff, 'Gf_eff': Gf.mean(),
        'vol_fracs': vol_fracs,
    }


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concrete Mesostructure Generator')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--size', type=float, default=50.0, help='Domain size (mm)')
    parser.add_argument('--vol_frac', type=float, default=0.45, help='Aggregate vol fraction')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='Voxel size (mm)')
    parser.add_argument('--output', type=str, default='concrete_meso.npz')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    spec = ConcreteSpec(
        Lx=args.size, Ly=args.size, Lz=args.size if args.dim == 3 else 0,
        voxel_size=args.voxel_size, vol_frac_agg=args.vol_frac)

    print(f"Generating {args.dim}D concrete mesostructure...")
    print(f"  Size: {args.size}mm, voxel: {args.voxel_size}mm, "
          f"agg: {args.vol_frac*100:.0f}%")

    if args.dim == 2:
        voxels, agg_info = generate_2d_concrete(spec)
    else:
        voxels, agg_info = generate_3d_concrete(spec)

    # Effective properties
    eff = homogenize_effective_properties(voxels, spec)
    print(f"\nEffective Properties:")
    print(f"  E_eff = {eff['E_eff']:.0f} MPa (Voigt={eff['E_voigt']:.0f}, "
          f"Reuss={eff['E_reuss']:.0f})")
    print(f"  ft_eff = {eff['ft_eff']:.2f} MPa")
    print(f"  Gf_eff = {eff['Gf_eff']:.4f} N/mm")
    print(f"  Phase fractions: {eff['vol_fracs']}")

    # Save
    np.savez_compressed(args.output, voxels=voxels,
                        agg_positions=np.array(agg_info['positions']),
                        agg_diameters=np.array(agg_info['diameters']),
                        spec=vars(spec), effective_props=eff)
    print(f"\nSaved: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")

    if args.plot and args.dim == 2:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = {1: 'gray', 2: 'white', 3: 'red'}
        cmap = plt.cm.colors.ListedColormap(['black', 'gray', 'white', 'red'])
        axes[0].imshow(voxels.T, origin='lower', cmap=cmap, vmin=0, vmax=3)
        axes[0].set_title('Mesostructure')
        axes[1].imshow((voxels == 2).T, origin='lower', cmap='gray_r')
        axes[1].set_title('Aggregates only')
        axes[2].imshow((voxels == 3).T, origin='lower', cmap='Reds')
        axes[2].set_title('ITZ only')
        plt.tight_layout()
        plt.savefig('concrete_meso.png', dpi=150)
        print("Plot saved: concrete_meso.png")
