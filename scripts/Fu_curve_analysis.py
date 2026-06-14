"""
F*-u* Load-Displacement Curve Analysis
=======================================
Derives the complete structural response from the 1D uPF-CZM analytical solution.

For a 1D bar (length 2L, area A, crack at center):
  F*(d*)  = sigma * A = ft * (1-d*)^(2p) * A
  u*(d*)  = u_elastic + w(0; d*) = sigma*2L/E + w(0; d*)

Key dimensionless parameter:
  lambda = l_ch / L = E*Gf / (ft^2 * L)  — brittleness number

Post-peak behavior:
  lambda < 1  -> snap-back (brittle)
  lambda = 1  -> vertical drop
  lambda > 1  -> gradual softening (ductile)
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from micro_pf_czm_1d import MicroPF_CZM_1D

OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Material parameters (concrete-like)
E, ft, Gf = 30000.0, 3.0, 0.12
A = 10000.0  # cross-sectional area (mm^2), e.g. 100mm x 100mm
l_ch = E * Gf / ft**2  # = 400 mm


def compute_Fu_curve(model, L_bar, d_star_vals=None):
    """
    Compute F*-u* curve for a 1D bar of half-length L_bar.

    Parameters
    ----------
    model : MicroPF_CZM_1D
        Material model
    L_bar : float
        Bar half-length (mm)
    d_star_vals : ndarray, optional
        Sequence of d* values

    Returns
    -------
    F_vals : ndarray — Force (N)
    u_vals : ndarray — Total displacement (mm)
    d_star_vals : ndarray
    sigma_vals : ndarray — Stress at each d*
    """
    if d_star_vals is None:
        d_star_vals = np.linspace(0.0, 0.999, 500)

    sigma_vals = model.ft * (1.0 - d_star_vals)**(2.0 * model.p)
    F_vals = sigma_vals * A

    # Elastic displacement: u_el = sigma * 2L / E
    u_elastic = sigma_vals * 2.0 * L_bar / E

    # Crack opening displacement at center
    w0_vals = np.array([model.compute_center_cod(d) for d in d_star_vals])

    # Total displacement
    u_vals = u_elastic + w0_vals

    return F_vals, u_vals, d_star_vals, sigma_vals


def plot_Fu_size_effect(model=None, output_dir=OUTPUT_DIR):
    """
    Plot F*-u* curves for different structural sizes (L values).

    Shows the ductile-to-brittle transition as bar length increases.
    This is the structural manifestation of the size effect.
    """
    if model is None:
        model = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=1.0)

    L_values = [25, 50, 100, 200, 400, 800, 1600]  # mm (half-lengths)
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(L_values)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- (a) F* vs u* ---
    ax = axes[0]
    for L_bar, color in zip(L_values, colors):
        lam = l_ch / L_bar
        F_vals, u_vals, _, _ = compute_Fu_curve(model, L_bar)

        label_str = f'$L={L_bar}$ mm ($\\lambda={lam:.1f}$)'
        if lam < 1:
            label_str += ' [brittle]'

        ax.plot(u_vals, F_vals / 1000, color=color, linewidth=2.0, label=label_str)

        # Mark peak
        peak_idx = np.argmax(F_vals)
        if L_bar >= 100:  # only mark for larger bars
            ax.scatter(u_vals[peak_idx], F_vals[peak_idx]/1000,
                      s=40, c=color, edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_xlabel('Displacement $u^*$ (mm)', fontsize=12)
    ax.set_ylabel('Force $F^*$ (kN)', fontsize=12)
    ax.set_title('(a) Load-Displacement Curves: Size Effect', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, framealpha=0.8, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, None)

    # --- (b) Normalized F/F_max vs u/u_peak ---
    ax = axes[1]
    for L_bar, color in zip(L_values, colors):
        lam = l_ch / L_bar
        F_vals, u_vals, _, _ = compute_Fu_curve(model, L_bar)

        F_norm = F_vals / F_vals.max()
        u_peak = u_vals[0]  # displacement at peak (d*~0)
        u_norm = u_vals / u_peak if u_peak > 0 else u_vals

        ax.plot(u_norm, F_norm, color=color, linewidth=2.0, label=f'$L={L_bar}$ ($\\lambda={lam:.1f}$)')

    # Snap-back threshold line
    ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(1.02, 0.5, '$u = u_{peak}$', rotation=90, va='center', fontsize=9, alpha=0.5)

    ax.set_xlabel('Normalized Displacement $u^* / u_{peak}$', fontsize=12)
    ax.set_ylabel('Normalized Force $F^* / F_{max}$', fontsize=12)
    ax.set_title('(b) Normalized Response', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        f'Structural Response of 1D Bar with Center Crack\n'
        f'($f_t={ft:.1f}$ MPa, $G_f={Gf:.3f}$ N/mm, '
        f'$l_{{ch}}={l_ch:.0f}$ mm, $p={model.p}$, linear softening)',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'Fu_size_effect.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[F-u] Size effect: {path}')
    return path


def plot_Fu_p_effect(model=None, L_bar=200, output_dir=OUTPUT_DIR):
    """
    Plot F*-u* curves for different traction orders p at fixed size.

    Shows how p affects the post-peak response.
    """
    if model is None:
        model = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=1.0)

    p_values = [1.0, 1.5, 2.0, 3.0]
    colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    lam = l_ch / L_bar

    for p_val, color in zip(p_values, colors_p):
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=p_val)
        F_vals, u_vals, d_star, sigma = compute_Fu_curve(m, L_bar)

        ax1.plot(u_vals, F_vals / 1000, color=color, linewidth=2.2,
                 label=f'$p={p_val}$')

        # Normalized
        F_norm = F_vals / F_vals.max()
        u_norm = u_vals / u_vals[0] if u_vals[0] > 0 else u_vals
        ax2.plot(u_norm, F_norm, color=color, linewidth=2.2,
                 label=f'$p={p_val}$')

    F_max = ft * A / 1000
    ax1.axhline(y=F_max, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Displacement $u^*$ (mm)', fontsize=12)
    ax1.set_ylabel('Force $F^*$ (kN)', fontsize=12)
    ax1.set_title(f'(a) F-u Curves ($L={L_bar}$ mm, $\\lambda={lam:.1f}$)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Normalized Displacement $u^* / u_{peak}$', fontsize=12)
    ax2.set_ylabel('Normalized Force $F^* / F_{max}$', fontsize=12)
    ax2.set_title(f'(b) Normalized ($L={L_bar}$ mm)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 1.05)

    fig.suptitle('Effect of Traction Order $p$ on Structural Response',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'Fu_p_effect.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[F-u] p-effect: {path}')
    return path


def plot_Fu_softening_effect(L_bar=200, output_dir=OUTPUT_DIR):
    """Plot F*-u* curves for different softening types."""
    softening_types = ['linear', 'exponential', 'cornelissen', 'ppr']
    colors_st = {'linear': '#1f77b4', 'exponential': '#ff7f0e',
                 'cornelissen': '#2ca02c', 'ppr': '#d62728'}
    ls_st = {'linear': '-', 'exponential': '--', 'cornelissen': '-.', 'ppr': ':'}
    labels = {'linear': 'Linear', 'exponential': 'Exponential',
              'cornelissen': 'Cornelissen', 'ppr': 'PPR'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    lam = l_ch / L_bar

    for st in softening_types:
        try:
            m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening=st, p=1.5)
            F_vals, u_vals, _, _ = compute_Fu_curve(m, L_bar)

            ax1.plot(u_vals, F_vals / 1000, color=colors_st[st],
                     ls=ls_st[st], linewidth=2.2, label=labels[st])

            F_norm = F_vals / F_vals.max()
            u_norm = u_vals / u_vals[0] if u_vals[0] > 0 else u_vals
            ax2.plot(u_norm, F_norm, color=colors_st[st],
                     ls=ls_st[st], linewidth=2.2, label=labels[st])
        except Exception as e:
            print(f'  [{st}] skipped: {e}')

    ax1.set_xlabel('Displacement $u^*$ (mm)', fontsize=12)
    ax1.set_ylabel('Force $F^*$ (kN)', fontsize=12)
    ax1.set_title(f'(a) F-u Curves ($L={L_bar}$ mm, $\\lambda={lam:.1f}$, $p=1.5$)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Normalized Displacement $u^* / u_{peak}$', fontsize=12)
    ax2.set_ylabel('Normalized Force $F^* / F_{max}$', fontsize=12)
    ax2.set_title(f'(b) Normalized ($L={L_bar}$ mm)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 1.05)

    fig.suptitle('Effect of Softening Type on Structural Response',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'Fu_softening_effect.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[F-u] Softening effect: {path}')
    return path


def plot_Fu_paper_validation(output_dir=OUTPUT_DIR):
    """
    Comprehensive validation: F*-u* curves with analytical verification.

    For linear softening with p=1:
      \bar{F} = (1-d*)^2
      \bar{u} = \bar{F} + \lambda * (1 - \bar{F})

    The post-peak branch is LINEAR in (F, u) space.
    """
    model = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=1.0)

    L_values = [50, 100, 200, 400, 800]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_values)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- (a) F*-u* curves for different L ---
    ax = axes[0, 0]
    for L_bar, color in zip(L_values, colors):
        lam = l_ch / L_bar
        F_vals, u_vals, d_star, _ = compute_Fu_curve(model, L_bar)

        # Numerical curve
        ax.plot(u_vals, F_vals / 1000, color=color, linewidth=2.2,
                label=f'$L={L_bar}$ mm ($\\lambda={lam:.1f}$)')

        # Analytical verification: \bar{F} = \bar{u}/(\lambda + (1-\lambda)\bar{u})^?
        # Actually: \bar{u} = \bar{F} + \lambda*(1-\bar{F}) for linear p=1
        # Let's verify by plotting the analytical relationship
        if L_bar >= 100:
            F_ana = np.linspace(F_vals[-1], F_vals[0], 100)
            F_norm_ana = F_ana / F_ana.max()
            u_elastic = F_ana * 2 * L_bar / (E * A)
            w_c = 2 * Gf / ft
            # From F_norm = (1-d*)^2: d* = 1 - sqrt(F_norm)
            d_star_ana = 1.0 - np.sqrt(F_norm_ana)
            w_ana = w_c * (2*d_star_ana - d_star_ana**2)
            u_ana = u_elastic + w_ana
            ax.plot(u_ana, F_ana / 1000, '--', color=color, linewidth=0.8, alpha=0.6)

    F_max = ft * A / 1000
    ax.axhline(y=F_max, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Displacement $u^*$ (mm)', fontsize=11)
    ax.set_ylabel('Force $F^*$ (kN)', fontsize=11)
    ax.set_title('(a) F*-u* Size Effect (dashed = analytical)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # --- (b) Peak load vs size (size effect law) ---
    ax = axes[0, 1]
    L_range = np.logspace(np.log10(20), np.log10(2000), 50)
    sigma_N_vals = []
    sigma_N_ana = []

    for L_bar in L_range:
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=1.0)
        detail = m.peak_load_detail(L_char=L_bar, width=100, height=100)
        sigma_N_vals.append(detail['sigma_N'])

        # Analytical: sigma_N = B*ft/sqrt(1+L_char/l_ch)  with B from the exact 1D solution
        # For the 1D bar: F_max = ft*A, sigma_N = ft (strength-controlled limit)
        # Actually the exact peak is just ft since there's no stress concentration
        sigma_N_ana.append(ft)

    ax.loglog(L_range, sigma_N_vals, 'b-', linewidth=2, label='PeakLoad (extended)')
    ax.loglog(L_range, sigma_N_ana, 'r--', linewidth=1.5, label='1D exact ($\\sigma_N=f_t$)')
    ax.set_xlabel('Characteristic Size $L$ (mm)', fontsize=11)
    ax.set_ylabel('Nominal Stress $\\sigma_N$ (MPa)', fontsize=11)
    ax.set_title('(b) Size Effect on Peak Load', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which='both')

    # --- (c) Fracture energy verification ---
    ax = axes[1, 0]
    L_test = 200
    for p_val, color in zip([1.0, 1.5, 2.0], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=p_val)
        F_vals, u_vals, _, sigma = compute_Fu_curve(m, L_test)

        # External work
        W_ext = np.trapz(F_vals, u_vals)

        # Elastic energy at peak
        u_peak = u_vals[0]
        W_elastic = 0.5 * F_vals[0] * u_peak

        # Fracture energy = external work - elastic energy released
        Gf_computed = (W_ext - W_elastic) / A

        # Plot force-displacement with area fill
        ax.fill_between(u_vals, 0, F_vals / 1000, alpha=0.1, color=color)
        ax.plot(u_vals, F_vals / 1000, color=color, linewidth=2,
                label=f'$p={p_val}$ ($G_f^{{num}}={Gf_computed:.4f}$)')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Displacement $u^*$ (mm)', fontsize=11)
    ax.set_ylabel('Force $F^*$ (kN)', fontsize=11)
    ax.set_title(f'(c) Fracture Energy Verification ($L={L_test}$ mm)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # --- (d) Ductility index vs brittleness ---
    ax = axes[1, 1]
    L_scan = np.logspace(np.log10(10), np.log10(2000), 100)
    for p_val, color in zip([1.0, 1.5, 2.0], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=p_val)
        ductility = []
        for L_bar in L_scan:
            F_vals, u_vals, _, _ = compute_Fu_curve(m, L_bar)
            u_peak = u_vals[0]
            u_final = u_vals[-1]
            ductility.append(u_final / u_peak if u_peak > 0 else 1.0)
        ax.semilogx(L_scan, ductility, color=color, linewidth=2, label=f'$p={p_val}$')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax.text(50, 1.1, 'Snap-back threshold', fontsize=8, alpha=0.5)
    ax.set_xlabel('Bar Half-Length $L$ (mm)', fontsize=11)
    ax.set_ylabel('Ductility Ratio $u_{final} / u_{peak}$', fontsize=11)
    ax.set_title('(d) Ductility vs Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, None)

    fig.suptitle(
        f'F*-u* Structural Response — Full Validation\n'
        f'($E={E/1000:.0f}$ GPa, $f_t={ft:.1f}$ MPa, $G_f={Gf:.3f}$ N/mm, '
        f'$l_{{ch}}={l_ch:.0f}$ mm, $A={A/100:.0f}$ cm$^2$)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'Fu_paper_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[F-u] Paper validation: {path}')
    return path


def plot_Fu_paper_style_summary(output_dir=OUTPUT_DIR):
    """
    Create a publication-style summary figure.
    Shows the complete structural response in the format
    closest to the paper's expected F*-u* presentation.
    """
    model = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=1.0)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # --- (a) F*-u* for multiple sizes ---
    ax_a = fig.add_subplot(gs[0, :2])
    L_sizes = [30, 60, 120, 240, 480, 960]
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(L_sizes)))
    for L_bar, color in zip(L_sizes, colors):
        lam = l_ch / L_bar
        F, u, _, _ = compute_Fu_curve(model, L_bar)
        # Mark snap-back region
        if lam < 1:
            # Find the snap-back point where du/dF changes sign
            dudF = np.gradient(u, F)
            snap_idx = np.where(dudF < 0)[0]
            if len(snap_idx) > 0:
                ax_a.plot(u[:snap_idx[0]], F[:snap_idx[0]]/1000, '--', color=color, linewidth=1.2, alpha=0.5)

        ax_a.plot(u, F/1000, color=color, linewidth=2.2,
                  label=f'$L={L_bar}$ mm, $\\lambda={lam:.2f}$' + ('*' if lam < 1 else ''))

    F_peak = ft * A / 1000
    ax_a.axhline(y=F_peak, color='black', linestyle=':', alpha=0.3, linewidth=1)
    ax_a.set_xlabel('Displacement $u^*$ (mm)', fontsize=11)
    ax_a.set_ylabel('Force $F^*$ (kN)', fontsize=11)
    ax_a.set_title('(a) Load-Displacement Curves ($p=1$, linear softening)', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=8, ncol=3, framealpha=0.8, loc='upper right')
    ax_a.grid(True, alpha=0.2)
    ax_a.text(0.02, 0.98, '* = snap-back regime', transform=ax_a.transAxes,
              fontsize=8, va='top', style='italic', color='gray')

    # --- (b) Normalized F*-u* (self-similar scaling) ---
    ax_b = fig.add_subplot(gs[0, 2])
    for L_bar, color in zip(L_sizes, colors):
        lam = l_ch / L_bar
        F, u, _, _ = compute_Fu_curve(model, L_bar)
        F_norm = F / F.max()
        u_peak = u[0]
        # Normalize by the characteristic displacement
        u_char = ft * 2 * L_bar / E + 2 * Gf / ft  # elastic + full COD
        u_norm = u / u_char
        ax_b.plot(u_norm, F_norm, color=color, linewidth=2.0,
                  label=f'$\\lambda={lam:.1f}$')

    ax_b.set_xlabel('Normalized Displacement', fontsize=11)
    ax_b.set_ylabel('Normalized Force $F/F_{max}$', fontsize=11)
    ax_b.set_title('(b) Normalized Response', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=7, ncol=2, framealpha=0.8)
    ax_b.grid(True, alpha=0.2)
    ax_b.set_xlim(0, 1.5)
    ax_b.set_ylim(0, 1.05)

    # --- (c) p-effect on F*-u* (fixed size) ---
    ax_c = fig.add_subplot(gs[1, 0])
    L_fixed = 200
    p_vals = [1.0, 1.5, 2.0, 3.0]
    colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for p_val, color in zip(p_vals, colors_p):
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=p_val)
        F, u, _, _ = compute_Fu_curve(m, L_fixed)
        ax_c.plot(u, F/1000, color=color, linewidth=2.2, label=f'$p={p_val}$')

    ax_c.axhline(y=F_peak, color='gray', linestyle=':', alpha=0.4)
    ax_c.set_xlabel('Displacement $u^*$ (mm)', fontsize=11)
    ax_c.set_ylabel('Force $F^*$ (kN)', fontsize=11)
    ax_c.set_title(f'(c) Effect of $p$ ($L={L_fixed}$ mm)', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.2)

    # --- (d) Softening effect on F*-u* (fixed size, p=1.5) ---
    ax_d = fig.add_subplot(gs[1, 1])
    for st, color, ls in [('linear', '#1f77b4', '-'), ('exponential', '#ff7f0e', '--'),
                           ('cornelissen', '#2ca02c', '-.'), ('ppr', '#d62728', ':')]:
        try:
            m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening=st, p=1.5)
            F, u, _, _ = compute_Fu_curve(m, L_fixed)
            ax_d.plot(u, F/1000, color=color, ls=ls, linewidth=2.2,
                      label=st.capitalize())
        except Exception:
            pass

    ax_d.set_xlabel('Displacement $u^*$ (mm)', fontsize=11)
    ax_d.set_ylabel('Force $F^*$ (kN)', fontsize=11)
    ax_d.set_title(f'(d) Effect of Softening ($L={L_fixed}$ mm, $p=1.5$)', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.2)

    # --- (e) Brittleness transition diagram ---
    ax_e = fig.add_subplot(gs[1, 2])
    L_grid = np.logspace(0.5, 3.5, 200)
    p_grid = [1.0, 1.5, 2.0, 3.0]

    for p_val in p_grid:
        m = MicroPF_CZM_1D(E=E, ft=ft, Gf=Gf, softening='linear', p=p_val)
        duct_ratio = []
        for Lb in L_grid:
            F, u, _, _ = compute_Fu_curve(m, Lb)
            duct_ratio.append(u[-1] / u[0] if u[0] > 0 else 1.0)
        ax_e.semilogx(L_grid, duct_ratio, linewidth=2, label=f'$p={p_val}$')

    ax_e.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_e.fill_between([1, 2000], 0, 1.0, alpha=0.05, color='red')
    ax_e.text(800, 0.5, 'Snap-back\nregime', fontsize=8, ha='center', color='red', alpha=0.6)
    ax_e.text(10, 3.0, 'Ductile\nregime', fontsize=8, ha='center', color='green', alpha=0.6)
    ax_e.set_xlabel('Bar Half-Length $L$ (mm)', fontsize=11)
    ax_e.set_ylabel('Ductility $u_{final} / u_{peak}$', fontsize=11)
    ax_e.set_title('(e) Brittleness Transition Diagram', fontsize=12, fontweight='bold')
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.2)
    ax_e.set_ylim(0, 8)

    fig.suptitle(
        f'Complete Structural Response of $\\mu$PF-CZM 1D Model\n'
        f'($E={E/1000:.0f}$ GPa, $f_t={ft:.1f}$ MPa, $G_f={Gf:.3f}$ N/mm, '
        f'$l_{{ch}}={l_ch:.0f}$ mm, $A={A/100:.0f}$ cm$^2$)',
        fontsize=12, fontweight='bold', y=1.01
    )
    path = os.path.join(output_dir, 'Fu_comprehensive_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[F-u] Comprehensive summary: {path}')
    return path


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  F*-u* Load-Displacement Curve Analysis')
    print(f'  l_ch = {l_ch:.0f} mm, F_max = {ft*A/1000:.1f} kN')
    print('=' * 60)
    print()

    print('[1/5] Size effect F*-u* curves...')
    plot_Fu_size_effect()

    print('[2/5] p-effect on F*-u* curves...')
    plot_Fu_p_effect()

    print('[3/5] Softening effect on F*-u* curves...')
    plot_Fu_softening_effect()

    print('[4/5] Full paper validation...')
    plot_Fu_paper_validation()

    print('[5/5] Publication-style summary figure...')
    plot_Fu_paper_style_summary()

    print()
    print('=' * 60)
    print(f'  All F*-u* curves saved to: {OUTPUT_DIR}/')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if 'Fu_' in f:
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f'    {f} ({size_kb:.1f} KB)')
    print('=' * 60)
