"""
SENB Three-Point Bending Benchmark
===================================
Runs the uPF-CZM 2D FEM solver for a Single-Edge-Notched Beam
and generates load-displacement curves.

The solver uses the EXACT paper formulation (Appendix D):
  - alpha(d) = 2d-d^2 (optimal geometric)
  - omega(d) = 1/(1+phi(d)) (degradation)
  - varpi'(d) = -omega^2*mu'(d) (dissipation)

Mesh size requirement (paper Section 5.2):
  h <= b/5 within the cracking sub-domain

For quick testing, use smaller b or coarser mesh.
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


def run_senb_simulation(D=50.0, a0_ratio=0.2, h_el=2.0,
                        E=30000.0, nu=0.2, ft=3.0, Gf=0.12,
                        softening='linear', p=1.5, b=4.0,
                        u_max=0.1, n_steps=40, n_stagger=30,
                        verbose=True):
    """
    Run a complete SENB three-point bending simulation.

    Parameters
    ----------
    D : float — beam depth (mm)
    a0_ratio : float — notch depth ratio
    h_el : float — target element size (mm), should be <= b/5
    E, nu, ft, Gf — material properties
    softening : str — softening type
    p : float — traction order
    b : float — phase-field length scale (mm)
    u_max : float — maximum prescribed displacement (mm)
    n_steps : int — number of load steps
    n_stagger : int — max stagger iterations per step
    verbose : bool — print progress

    Returns
    -------
    dict with 'u_history', 'F_history', 'd_max_history', 'solver', 'mesh'
    """
    t_start = time.time()

    # --- Mesh ---
    if verbose:
        print(f"Generating mesh: D={D}mm, notch={a0_ratio*D:.0f}mm, h={h_el}mm")
    mesh, notch_f, bp = generate_senb_mesh_tri(
        D=D, a0_ratio=a0_ratio, h_el=h_el)
    n_nodes = mesh.p.shape[1]
    n_elem = mesh.t.shape[1]
    if verbose:
        print(f"  Mesh: {n_nodes} nodes, {n_elem} elements")

    # --- Material ---
    mat = MuPFCZMMaterial(E=E, nu=nu, Gf=Gf, ft=ft,
                          softening=softening, p=p, b=b)
    if verbose:
        print(f"  Material: E={E} MPa, ft={ft} MPa, Gf={Gf} N/mm")
        print(f"  lch={mat.lch:.0f} mm, a0={mat.a0:.1f}, b={b} mm, p={p}")
        print(f"  h/b ratio = {h_el/b:.2f} (need <= 0.2 for accuracy)")

    # --- Solver ---
    solver = MuPFCZM2DSolver(mat, mesh)

    # Initial damage on notch
    d_init = np.zeros(solver.ndof_d)
    set_notch_initial_damage(mesh, notch_f, d_init)
    solver.set_initial_damage(d_init)

    # --- Boundary conditions ---
    bc = find_senb_dofs(mesh, bp)
    constrained = (list(bc['supp_ux']) + list(bc['supp_uy']) +
                   list(bc['supp_uy_r']) + list(bc['load_uy']))
    constrained_list = list(set(constrained))
    free = [i for i in range(solver.ndof_u) if i not in set(constrained)]

    if verbose:
        print(f"  BC: {len(constrained_list)} constrained DOFs, "
              f"{len(free)} free DOFs")
        print(f"  Loading DOFs: {len(bc['load_uy'])}")

    # --- Loading history ---
    u_vals = np.linspace(0, u_max, n_steps + 1)[1:]  # skip zero
    history = {'u': [], 'F': [], 'd_max': [], 'steps': []}

    if verbose:
        print(f"\n  Running {n_steps} load steps (u_max={u_max} mm)...")

    for i, u_bar in enumerate(u_vals):
        # Reverse sign for downward loading
        u_new, d_new, F, conv = solver.solve_step(
            -u_bar,
            loading_dofs=bc['load_uy'],
            support_dofs=(bc['supp_ux'] + bc['supp_uy'] + bc['supp_uy_r']),
            n_stagger=n_stagger, tol=1e-3)

        u_history = -u_bar  # store as positive downward
        history['u'].append(u_history)
        history['d_max'].append(d_new.max())

        # Compute reaction force at loading point
        K_u = solver.assemble_degraded_stiffness(d_new)
        reaction = K_u @ u_new
        F_load = abs(reaction[bc['load_uy']].sum()) / 1000.0  # kN
        history['F'].append(F_load)

        if verbose and (i % max(1, n_steps//8) == 0 or conv == False):
            print(f"    Step {i+1:3d}/{n_steps}: u={-u_bar:.4f} mm, "
                  f"F={F_load:.4f} kN, d_max={d_new.max():.4f}, "
                  f"{'conv' if conv else 'NC'}")

        if not conv:
            if verbose:
                print(f"    [!] Stagger did not converge at step {i+1}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s ({elapsed/n_steps:.2f}s/step)")

    return {
        'u_history': np.array(history['u']),
        'F_history': np.array(history['F']),
        'd_max_history': np.array(history['d_max']),
        'solver': solver,
        'mesh': mesh,
        'beam_params': bp,
        'bc': bc,
    }


def plot_results(result, output_dir='paper_figures', label=''):
    """Plot F-u curve, damage evolution, and deformed shape."""
    os.makedirs(output_dir, exist_ok=True)

    u = result['u_history']
    F = result['F_history']
    d_max = result['d_max_history']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Load-displacement
    ax = axes[0]
    ax.plot(u, F, 'b-', linewidth=2)
    ax.set_xlabel('Displacement u (mm)', fontsize=11)
    ax.set_ylabel('Force F (kN)', fontsize=11)
    ax.set_title('Load-Displacement Curve', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Damage evolution
    ax = axes[1]
    ax.plot(u, d_max, 'r-', linewidth=2)
    ax.set_xlabel('Displacement u (mm)', fontsize=11)
    ax.set_ylabel('Max Damage d_max', fontsize=11)
    ax.set_title('Damage Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) F vs d_max
    ax = axes[2]
    ax.plot(d_max, F, 'g-', linewidth=2)
    ax.set_xlabel('Max Damage d_max', fontsize=11)
    ax.set_ylabel('Force F (kN)', fontsize=11)
    ax.set_title('Force vs Damage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    suffix = f'_{label}' if label else ''
    plt.suptitle(f'SENB Three-Point Bending — uPF-CZM {label}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'senb_results{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Results saved: {path}")
    return path


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  SENB Benchmark — uPF-CZM 2D FEM")
    print("=" * 60)

    # Quick test with moderate resolution
    # h/b = 0.8/4 = 0.2 (at the limit of acceptable accuracy)
    print("\n[Test 1] D=50mm, coarse mesh, linear softening, p=1.5")
    r1 = run_senb_simulation(
        D=50.0, a0_ratio=0.2, h_el=1.5,
        E=30000.0, nu=0.2, ft=3.0, Gf=0.12,
        softening='linear', p=1.5, b=5.0,
        u_max=0.08, n_steps=30, n_stagger=30,
        verbose=True)

    plot_results(r1, label='D50_linear_p1.5')

    # Overall summary
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print(f"  F_max = {r1['F_history'].max():.4f} kN")
    print(f"  u_at_F_max = {r1['u_history'][np.argmax(r1['F_history'])]:.4f} mm")
    print(f"  Final d_max = {r1['d_max_history'][-1]:.4f}")
    print("=" * 60)
