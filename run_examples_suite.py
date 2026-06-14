"""
uPF-CZM Numerical Examples Suite
=================================
Multiple benchmark cases with Paraview VTK export.

Examples:
  Ex.1 — SENB: size effect (D=15, 25, 40mm)
  Ex.2 — SENB: softening type comparison
  Ex.3 — SENB: p-dependence study
  Ex.4 — Single element test (validation)
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from fem_2d import MuPFCZMMaterial, MuPFCZM2DSolver, find_senb_dofs
from mesh_utils import generate_senb_mesh_tri, set_notch_initial_damage
from vtk_export import export_vtk, export_vtk_sequence

OUTPUT_DIR = 'examples_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
VTK_DIR = os.path.join(OUTPUT_DIR, 'vtk')
os.makedirs(VTK_DIR, exist_ok=True)


def run_one_case(name, D, a0_ratio, h_el, E, nu, ft, Gf, softening, p, b,
                 u_max, n_steps, n_stagger=6):
    """Run one SENB case and return results."""
    print(f'\n{"="*50}')
    print(f'  {name}')
    print(f'{"="*50}')

    lch = E * Gf / ft**2
    lam = lch / D
    a0_val = (2*p/np.pi) * lch / b
    print(f'  D={D}mm, lch={lch:.0f}mm, lambda={lam:.1f}, a0={a0_val:.1f}')
    print(f'  h={h_el}mm, b={b}mm, h/b={h_el/b:.2f}')

    t0 = time.time()

    # Mesh
    mesh, nf, bp = generate_senb_mesh_tri(D=D, a0_ratio=a0_ratio, h_el=h_el)
    n_nodes = mesh.p.shape[1]
    n_elems = mesh.t.shape[1]
    print(f'  Mesh: {n_nodes} nodes, {n_elems} elems')

    # Material + Solver
    mat = MuPFCZMMaterial(E=E, nu=nu, Gf=Gf, ft=ft,
                          softening=softening, p=p, b=b)
    solver = MuPFCZM2DSolver(mat, mesh)
    di = np.zeros(solver.ndof_d)
    set_notch_initial_damage(mesh, nf, di, b=b, smooth=True)
    solver.set_initial_damage(di)

    # BC
    bc = find_senb_dofs(mesh, bp)
    supp = bc['supp_ux'] + bc['supp_uy'] + bc['supp_uy_r']
    load = bc['load_uy']

    # Loading
    u_vals = np.linspace(u_max/n_steps, u_max, n_steps)
    u_history = []
    F_history = []
    d_history = []

    for i, u_bar in enumerate(u_vals):
        ub = -u_bar
        t1 = time.time()
        u_new, d_new, _, conv = solver.solve_step(
            ub, load, supp, n_stagger=n_stagger, tol=1e-3)
        Ku = solver.assemble_degraded_stiffness(d_new)
        F = abs((Ku @ u_new)[load].sum()) / 1000
        u_history.append(u_bar)
        F_history.append(F)
        d_history.append(d_new.copy())

        if i % max(1, n_steps//4) == 0 or i == n_steps-1:
            dt = time.time() - t1
            nd5 = (d_new > 0.5).sum()
            print(f'    Step {i+1:2d}/{n_steps}: u={u_bar:.4f}mm '
                  f'F={F:.4f}kN d_max={d_new.max():.3f} '
                  f'd>0.5:{nd5}/{len(d_new)} [{dt:.1f}s]')

    elapsed = time.time() - t0
    print(f'  Completed in {elapsed:.1f}s')

    # Export VTK
    vtk_prefix = name.replace(' ', '_').replace('/', '_')
    export_vtk_sequence(solver, mesh, u_history, d_history,
                        output_dir=os.path.join(VTK_DIR, vtk_prefix),
                        prefix='step')

    return {
        'name': name,
        'u': np.array(u_history),
        'F': np.array(F_history),
        'd_max': np.array([d.max() for d in d_history]),
        'mesh': mesh,
        'solver': solver,
        'params': dict(D=D, E=E, ft=ft, Gf=Gf, softening=softening, p=p, b=b,
                       lch=lch, lam=lam, a0=a0_val),
    }


def plot_all_results(results, filename='all_examples.png'):
    """Plot F-u curves for all examples on one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # Group by example type
    groups = {}
    for r in results:
        key = r['name'].split(' (')[0] if ' (' in r['name'] else r['name']
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(g) for g in groups.values())))

    for ax_idx, (group_name, group_results) in enumerate(groups.items()):
        if ax_idx >= 4:
            break
        ax = axes[ax_idx]
        for j, r in enumerate(group_results):
            c = colors[j % len(colors)]
            p = r['params']
            label = (f"D={p['D']}mm" if 'size' in group_name.lower()
                     else f"{p['softening']}" if 'soften' in group_name.lower()
                     else f"p={p['p']}" if 'p=' in r['name'].lower()
                     else r['name'].split('(')[-1].rstrip(')') if '(' in r['name']
                     else '')
            ax.plot(r['u'], r['F'], color=c, linewidth=2, label=label)
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Force (kN)')
        ax.set_title(group_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('uPF-CZM Numerical Examples — Load-Displacement Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSummary plot saved: {path}')
    return path


# ====================================================================
# Example Definitions
# ====================================================================
EXAMPLES = [
    # --- Ex.1: Size effect (same material, different D) ---
    dict(name='Size Effect (D=15mm)', D=15, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=1.5, b=3.0, u_max=0.06, n_steps=12, n_stagger=8),
    dict(name='Size Effect (D=25mm)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=1.5, b=3.0, u_max=0.10, n_steps=12, n_stagger=8),
    dict(name='Size Effect (D=40mm)', D=40, a0_ratio=0.2, h_el=2.0,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=1.5, b=4.0, u_max=0.15, n_steps=10, n_stagger=8),

    # --- Ex.2: p-dependence ---
    dict(name='p-Dependence (p=1.0)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=1.0, b=3.0, u_max=0.10, n_steps=12, n_stagger=8),
    dict(name='p-Dependence (p=2.0)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=2.0, b=3.0, u_max=0.10, n_steps=12, n_stagger=8),

    # --- Ex.3: Softening type ---
    dict(name='Softening (Linear)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='linear',
         p=1.5, b=3.0, u_max=0.10, n_steps=12, n_stagger=8),
    dict(name='Softening (Exponential)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='exponential',
         p=1.5, b=3.0, u_max=0.12, n_steps=12, n_stagger=8),
    dict(name='Softening (Cornelissen)', D=25, a0_ratio=0.2, h_el=1.5,
         E=5000, nu=0.2, ft=3.0, Gf=0.12, softening='cornelissen',
         p=1.5, b=3.0, u_max=0.12, n_steps=12, n_stagger=8),
]


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  uPF-CZM Numerical Examples Suite')
    print(f'  {len(EXAMPLES)} cases configured')
    print(f'  Output: {OUTPUT_DIR}/')
    print(f'  Paraview VTK: {VTK_DIR}/')
    print('=' * 60)

    all_results = []
    for i, ex in enumerate(EXAMPLES):
        print(f'\n[{i+1}/{len(EXAMPLES)}] {ex["name"]}')
        try:
            r = run_one_case(**ex)
            all_results.append(r)
        except Exception as e:
            print(f'  [!] FAILED: {e}')

    # Summary plot
    if all_results:
        plot_all_results(all_results)

    # Print summary table
    print('\n' + '=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print(f'  {"Case":<35s} {"F_max(kN)":>10s} {"u@Fmax(mm)":>12s}')
    print('  ' + '-' * 57)
    for r in all_results:
        idx = np.argmax(r['F'])
        print(f'  {r["name"]:<35s} {r["F"][idx]:>10.4f} {r["u"][idx]:>12.5f}')
    print('=' * 60)
    print(f'\n  VTK files for Paraview: {VTK_DIR}/')
    print(f'  Open .pvd files in Paraview for animation.')
    print('=' * 60)
