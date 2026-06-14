"""
Reproduce Paper Figures 11 & 12 + Generate Fracture Animation
==============================================================
Reference: "A generalized phase-field cohesive zone model (uPF-CZM) for fracture"

Fig.11: (a) Phase-field crack profiles d(x) for different d*
        (b) Crack opening displacement profiles w(x)
Fig.12: (a) Traction-separation laws for different softening types
        (b) Center COD w(0) vs phase-field d* relation
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from micro_pf_czm_1d import MicroPF_CZM_1D

# ================================================================
# Global settings
# ================================================================
MATERIAL_PARAMS = dict(E=30000, ft=3.0, Gf=0.12, b=2.0)
OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = plt.cm.viridis(np.linspace(0.15, 0.95, 9))
D_STAR_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ================================================================
# FIGURE 11: Phase-field profiles and COD profiles
# ================================================================
def plot_figure_11(model=None, d_star_list=None, output_dir=OUTPUT_DIR):
    """
    Fig.11: Crack phase-field profiles d(x) and COD profiles w(x).

    Left (a): Phase-field d(x) for different center values d*
    Right (b): COD w(x) for the same d* values
    """
    if model is None:
        model = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=1.0)
    if d_star_list is None:
        d_star_list = D_STAR_LIST

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, d_star in enumerate(d_star_list):
        color = COLORS[i]

        # --- Phase-field profile d(x) ---
        x_d, d_vals = model.compute_d_profile(d_star)
        ax1.plot(x_d, d_vals, color=color, linewidth=1.8,
                 label=f'$d^*={d_star:.1f}$')
        ax1.fill_between(x_d, 0, d_vals, alpha=0.08, color=color)

        # --- COD profile w(x) ---
        x_w, w_vals = model.compute_cod_profile(d_star)
        ax2.plot(x_w, w_vals, color=color, linewidth=1.8,
                 label=f'$d^*={d_star:.1f}$')
        ax2.fill_between(x_w, 0, w_vals, alpha=0.08, color=color)

    # Axis labels and styling
    ax1.set_xlabel('Position $x$ (mm)', fontsize=12)
    ax1.set_ylabel('Phase-field $d(x)$', fontsize=12)
    ax1.set_title('(a) Crack Phase-Field Profiles', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, model.crack_half_width(0.95) * 1.05)
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.8)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Position $x$ (mm)', fontsize=12)
    ax2.set_ylabel('Crack Opening $w(x)$ (mm)', fontsize=12)
    ax2.set_title('(b) Crack Opening Displacement Profiles', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, model.crack_half_width(0.95) * 1.2)
    ax2.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.8)
    ax2.grid(True, alpha=0.2)

    # Parameter annotation
    w_c = 2 * model.Gf / model.ft
    fig.suptitle(
        f'Fig. 11: 1D Crack Profiles ($E={model.E/1000:.0f}$ GPa, '
        f'$f_t={model.ft:.1f}$ MPa, $G_f={model.Gf:.3f}$ N/mm, '
        f'$p={model.p}$, $b={model.b}$ mm, $w_c={w_c:.4f}$ mm)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig11_crack_profiles.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Fig.11] Saved: {path}')
    return path


# ================================================================
# FIGURE 12: Traction-separation laws and COD-d* relation
# ================================================================
def plot_figure_12(output_dir=OUTPUT_DIR):
    """
    Fig.12: Traction-separation laws and center COD vs d*.

    Left (a): sigma(w) / ft vs w / w_c for different softening types
    Right (b): w(0) / w_c vs d* for different softening types
    """
    softening_types = ['linear', 'exponential', 'cornelissen', 'ppr']
    colors_st = {'linear': '#1f77b4', 'exponential': '#ff7f0e',
                 'cornelissen': '#2ca02c', 'ppr': '#d62728'}
    linestyles = {'linear': '-', 'exponential': '--',
                  'cornelissen': '-.', 'ppr': ':'}
    labels = {'linear': 'Linear', 'exponential': 'Exponential',
              'cornelissen': 'Cornelissen', 'ppr': 'PPR ($m=1.5$)'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    d_star_fine = np.linspace(0.001, 0.999, 200)

    for st in softening_types:
        try:
            model = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening=st, p=1.5)
        except Exception:
            continue

        w_c = 2 * model.Gf / model.ft
        color = colors_st[st]
        ls = linestyles[st]

        # --- (a) Traction-separation law ---
        w_vals, sigma_vals = model.analytical_traction_separation(n_points=300)
        ax1.plot(w_vals / w_c, sigma_vals / model.ft, color=color,
                 linestyle=ls, linewidth=2.2, label=labels.get(st, st))

        # --- (b) Center COD vs d* ---
        w0_vals = np.array([model.compute_center_cod(d) for d in d_star_fine])
        ax2.plot(d_star_fine, w0_vals / w_c, color=color,
                 linestyle=ls, linewidth=2.2, label=labels.get(st, st))

    # (a) axis labels
    ax1.set_xlabel('Normalized COD $w / w_c$', fontsize=12)
    ax1.set_ylabel('Normalized Traction $\\sigma / f_t$', fontsize=12)
    ax1.set_title('(a) Traction-Separation Laws', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 5.5)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10, framealpha=0.8)
    ax1.grid(True, alpha=0.2)

    # (b) axis labels
    ax2.set_xlabel('Center Phase-Field $d^*$', fontsize=12)
    ax2.set_ylabel('Normalized Center COD $w(0) / w_c$', fontsize=12)
    ax2.set_title('(b) Center COD vs Phase-Field Relation', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(
        f'Fig. 12: Cohesive Zone Characteristics ($f_t={model.ft:.1f}$ MPa, '
        f'$G_f={model.Gf:.3f}$ N/mm, $p=1.5$)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig12_cohesive_laws.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Fig.12] Saved: {path}')
    return path


# ================================================================
# FIGURE 11+12 COMBINED (论文风格)
# ================================================================
def plot_figure_combined(output_dir=OUTPUT_DIR):
    """
    Combined 4-panel figure matching the paper's Fig.11 and Fig.12 layout.
    """
    model = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=1.0)
    softening_types = ['linear', 'exponential', 'cornelissen', 'ppr']
    colors_st = {'linear': '#1f77b4', 'exponential': '#ff7f0e',
                 'cornelissen': '#2ca02c', 'ppr': '#d62728'}
    ls_st = {'linear': '-', 'exponential': '--', 'cornelissen': '-.', 'ppr': ':'}
    labels = {'linear': 'Linear', 'exponential': 'Exponential',
              'cornelissen': 'Cornelissen', 'ppr': 'PPR'}

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- (a) Phase-field profiles ---
    ax_a = fig.add_subplot(gs[0, 0])
    for i, d_star in enumerate(D_STAR_LIST):
        x_d, d_vals = model.compute_d_profile(d_star)
        ax_a.plot(x_d, d_vals, color=COLORS[i], linewidth=1.8,
                  label=f'$d^*={d_star:.1f}$')
    ax_a.set_xlabel('Position $x$ (mm)', fontsize=11)
    ax_a.set_ylabel('Phase-field $d(x)$', fontsize=11)
    ax_a.set_title('(a) Crack Phase-Field Profiles', fontsize=12, fontweight='bold')
    ax_a.set_xlim(0, model.crack_half_width(0.95) * 1.05)
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)
    ax_a.grid(True, alpha=0.2)

    # --- (b) COD profiles ---
    ax_b = fig.add_subplot(gs[0, 1])
    for i, d_star in enumerate(D_STAR_LIST):
        x_w, w_vals = model.compute_cod_profile(d_star)
        ax_b.plot(x_w, w_vals, color=COLORS[i], linewidth=1.8,
                  label=f'$d^*={d_star:.1f}$')
    ax_b.set_xlabel('Position $x$ (mm)', fontsize=11)
    ax_b.set_ylabel('COD $w(x)$ (mm)', fontsize=11)
    ax_b.set_title('(b) Crack Opening Displacement Profiles', fontsize=12, fontweight='bold')
    ax_b.set_xlim(0, model.crack_half_width(0.95) * 1.3)
    ax_b.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)
    ax_b.grid(True, alpha=0.2)

    # --- (c) Traction-separation laws ---
    ax_c = fig.add_subplot(gs[1, 0])
    for st in softening_types:
        m = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening=st, p=1.5)
        w_c = 2 * m.Gf / m.ft
        wv, sv = m.analytical_traction_separation(300)
        ax_c.plot(wv / w_c, sv / m.ft, color=colors_st[st],
                  ls=ls_st[st], linewidth=2.2, label=labels[st])
    ax_c.set_xlabel('Normalized COD $w / w_c$', fontsize=11)
    ax_c.set_ylabel('Normalized Traction $\\sigma / f_t$', fontsize=11)
    ax_c.set_title('(c) Traction-Separation Laws', fontsize=12, fontweight='bold')
    ax_c.set_xlim(0, 5.5)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=9, framealpha=0.8)
    ax_c.grid(True, alpha=0.2)

    # --- (d) Center COD vs d* ---
    ax_d = fig.add_subplot(gs[1, 1])
    d_fine = np.linspace(0.001, 0.999, 200)
    for st in softening_types:
        m = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening=st, p=1.5)
        w_c = 2 * m.Gf / m.ft
        w0s = np.array([m.compute_center_cod(d) for d in d_fine])
        ax_d.plot(d_fine, w0s / w_c, color=colors_st[st],
                  ls=ls_st[st], linewidth=2.2, label=labels[st])
    ax_d.set_xlabel('Center Phase-Field $d^*$', fontsize=11)
    ax_d.set_ylabel('Normalized Center COD $w(0)/w_c$', fontsize=11)
    ax_d.set_title('(d) Center COD vs Phase-Field', fontsize=12, fontweight='bold')
    ax_d.set_xlim(0, 1)
    ax_d.legend(fontsize=9, framealpha=0.8)
    ax_d.grid(True, alpha=0.2)

    w_c = 2 * model.Gf / model.ft
    fig.suptitle(
        f'Replication of Paper Figures 11 & 12\n'
        f'($E={model.E/1000:.0f}$ GPa, $f_t={model.ft:.1f}$ MPa, '
        f'$G_f={model.Gf:.3f}$ N/mm, $b={model.b}$ mm, '
        f'$w_c={w_c:.4f}$ mm, $l_{{ch}}={model.lch:.0f}$ mm)',
        fontsize=11, y=1.01
    )
    path = os.path.join(output_dir, 'fig11_12_combined.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Combined] Saved: {path}')
    return path


# ================================================================
# ANIMATION: Crack Evolution Video
# ================================================================
def create_crack_evolution_animation(
    model=None, output_dir=OUTPUT_DIR, fps=15, duration=12
):
    """
    Create MP4 animation of crack evolution from d*=0.01 to d*=0.99.

    Shows:
      - Left: d(x) profile evolution
      - Center: w(x) COD profile evolution
      - Right: sigma-w traction-separation trajectory
    """
    if model is None:
        model = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=1.0)

    n_frames = fps * duration
    d_star_seq = np.linspace(0.02, 0.98, n_frames)

    # Pre-compute profiles for frame efficiency
    print(f'[Animation] Pre-computing {n_frames} frames...')
    profiles = []
    for d_star in d_star_seq:
        x_d, d_vals = model.compute_d_profile(d_star, n_points=300)
        x_w, w_vals = model.compute_cod_profile(d_star, n_points=300)
        w0 = model.compute_center_cod(d_star)
        sigma_peak = model.ft * (1.0 - d_star)**(2.0 * model.p)
        profiles.append({
            'd_star': d_star, 'x_d': x_d, 'd_vals': d_vals,
            'x_w': x_w, 'w_vals': w_vals, 'w0': w0,
            'sigma': sigma_peak,
        })

    # Full traction-separation curve
    w_ts, sigma_ts = model.analytical_traction_separation(300)
    w_c = 2 * model.Gf / model.ft

    # Set up figure
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.1])

    ax_d = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1])
    ax_ts = fig.add_subplot(gs[2])

    x_max_d = model.crack_half_width(0.98) * 1.05
    x_max_w = model.crack_half_width(0.98) * 1.3

    def animate(frame_idx):
        """Update function for each animation frame."""
        prof = profiles[frame_idx]
        d_star = prof['d_star']

        # --- Clear axes ---
        ax_d.clear()
        ax_w.clear()
        ax_ts.clear()

        # --- (a) Phase-field profile ---
        ax_d.plot(prof['x_d'], prof['d_vals'], 'b-', linewidth=2.5)
        ax_d.fill_between(prof['x_d'], 0, prof['d_vals'], alpha=0.15, color='blue')
        ax_d.set_xlim(0, x_max_d)
        ax_d.set_ylim(0, 1.05)
        ax_d.set_xlabel('Position $x$ (mm)', fontsize=10)
        ax_d.set_ylabel('Phase-field $d(x)$', fontsize=10)
        ax_d.set_title(f'Phase-Field Profile ($d^*={d_star:.3f}$)', fontsize=11, fontweight='bold')
        ax_d.grid(True, alpha=0.2)
        ax_d.axhline(y=d_star, color='red', linestyle=':', alpha=0.4)
        ax_d.text(0.98, 0.95, f'$d^*={d_star:.3f}$', transform=ax_d.transAxes,
                  ha='right', va='top', fontsize=11, color='red',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # --- (b) COD profile ---
        ax_w.plot(prof['x_w'], prof['w_vals'], 'r-', linewidth=2.5)
        ax_w.fill_between(prof['x_w'], 0, prof['w_vals'], alpha=0.15, color='red')
        ax_w.set_xlim(0, x_max_w)
        ax_w.set_ylim(0, w_c * 1.1)
        ax_w.set_xlabel('Position $x$ (mm)', fontsize=10)
        ax_w.set_ylabel('COD $w(x)$ (mm)', fontsize=10)
        w0_val = prof['w0']
        w0_str = f'{w0_val:.4f}'
        ax_w.set_title('COD Profile ($w(0)=' + w0_str + '$ mm)', fontsize=11, fontweight='bold')
        ax_w.grid(True, alpha=0.2)
        ax_w.axhline(y=prof['w0'], color='red', linestyle=':', alpha=0.4)
        ax_w.text(0.98, 0.95, f'$w(0)={w0_str}$ mm', transform=ax_w.transAxes,
                  ha='right', va='top', fontsize=11, color='red',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # --- (c) Traction-separation trajectory ---
        ax_ts.plot(w_ts / w_c, sigma_ts / model.ft, 'gray', linewidth=1.5, alpha=0.5, label='TS Law')
        # Moving point
        ax_ts.scatter([prof['w0'] / w_c], [prof['sigma'] / model.ft],
                      s=150, c='red', zorder=10, edgecolors='darkred', linewidths=1.5)
        ax_ts.set_xlabel('Normalized COD $w / w_c$', fontsize=10)
        ax_ts.set_ylabel('Normalized Traction $\\sigma / f_t$', fontsize=10)
        ax_ts.set_title('Traction-Separation Trajectory', fontsize=11, fontweight='bold')
        ax_ts.set_xlim(0, 1.1)
        ax_ts.set_ylim(0, 1.05)
        ax_ts.legend(loc='upper right', fontsize=8)
        ax_ts.grid(True, alpha=0.2)

        # Progress annotation
        progress = frame_idx / (n_frames - 1) * 100
        ax_ts.text(0.5, -0.18, f'Crack Evolution: {progress:.0f}%',
                   transform=ax_ts.transAxes, ha='center', fontsize=10,
                   style='italic', color='gray')

        return []

    # Create animation
    print(f'[Animation] Generating {n_frames} frames...')
    ani = FuncAnimation(fig, animate, frames=n_frames,
                        interval=1000/fps, blit=False)

    # Save as MP4
    w_c_val = 2 * model.Gf / model.ft
    fig.suptitle(
        f'Crack Evolution in 1D $\\mu$PF-CZM '
        f'($f_t={model.ft:.1f}$ MPa, $G_f={model.Gf:.3f}$ N/mm, '
        f'$b={model.b}$ mm, $w_c={w_c_val:.4f}$ mm)',
        fontsize=11, y=1.02
    )

    mp4_path = os.path.join(output_dir, 'crack_evolution.mp4')
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='uPF-CZM'), bitrate=2000)
        ani.save(mp4_path, writer=writer, dpi=150)
        print(f'[Animation] MP4 Saved: {mp4_path}')
    except Exception as e:
        print(f'[Animation] FFmpeg not available ({e}), saving GIF instead...')
        gif_path = os.path.join(output_dir, 'crack_evolution.gif')
        ani.save(gif_path, writer='pillow', fps=fps, dpi=120)
        print(f'[Animation] GIF Saved: {gif_path}')
        mp4_path = gif_path

    plt.close()
    return mp4_path


# ================================================================
# SUPPLEMENTARY: Softening curve parameter study
# ================================================================
def plot_softening_comparison(output_dir=OUTPUT_DIR):
    """Compare traction-separation for all softening types at different p values."""
    softening_types = ['linear', 'exponential', 'cornelissen', 'ppr']
    p_values = [1.0, 1.5, 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for j, p_val in enumerate(p_values):
        ax = axes[j]
        for st in softening_types:
            try:
                m = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening=st, p=p_val)
                w_c = 2 * m.Gf / m.ft
                wv, sv = m.analytical_traction_separation(200)
                ax.plot(wv / w_c, sv / m.ft, linewidth=2, label=st.capitalize())
            except Exception:
                pass
        ax.set_xlabel('$w / w_c$', fontsize=11)
        ax.set_title(f'$p = {p_val}$', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('Normalized Traction $\\sigma / f_t$', fontsize=12)
    fig.suptitle('Softening Curve Comparison: Effect of $p$ and Softening Type',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'supp_softening_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Supp] Saved: {path}')
    return path


# ================================================================
# SUPPLEMENTARY: Energy dissipation evolution
# ================================================================
def plot_energy_evolution(model=None, output_dir=OUTPUT_DIR):
    """Plot energy dissipation as a function of d*."""
    if model is None:
        model = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=1.0)

    d_stars = np.linspace(0.01, 0.99, 100)
    w0_vals = np.array([model.compute_center_cod(d) for d in d_stars])
    sigma_vals = model.ft * (1.0 - d_stars)**(2.0 * model.p)

    # Energy dissipation = integral of sigma(w) from 0 to w(0)
    Gf_cumulative = np.zeros_like(d_stars)
    for i in range(1, len(d_stars)):
        dw = w0_vals[i] - w0_vals[i-1]
        Gf_cumulative[i] = Gf_cumulative[i-1] + (sigma_vals[i] + sigma_vals[i-1]) / 2 * dw

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(d_stars, Gf_cumulative, 'b-', linewidth=2)
    ax1.axhline(y=model.Gf, color='r', linestyle='--', label=f'$G_f={model.Gf:.4f}$ N/mm')
    ax1.set_xlabel('Center Phase-Field $d^*$', fontsize=11)
    ax1.set_ylabel('Cumulative Energy Dissipation (N/mm)', fontsize=11)
    ax1.set_title('Energy Dissipation Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Energy dissipation rate
    dG_dd = np.gradient(Gf_cumulative, d_stars)
    ax2.plot(d_stars, dG_dd, 'r-', linewidth=2)
    ax2.set_xlabel('Center Phase-Field $d^*$', fontsize=11)
    ax2.set_ylabel('$dG_f / dd^*$ (N/mm)', fontsize=11)
    ax2.set_title('Energy Dissipation Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'supp_energy_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Supp] Saved: {path}')
    return path


# ================================================================
# SUPPLEMENTARY: p-effect on crack profiles
# ================================================================
def plot_p_effect_profiles(output_dir=OUTPUT_DIR):
    """Show how p affects d(x) and w(x) profiles at fixed d*."""
    p_values = [1.0, 1.5, 2.0, 3.0]
    d_star_fixed = 0.7
    colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for p_val, color in zip(p_values, colors_p):
        m = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=p_val)

        x_d, d_vals = m.compute_d_profile(d_star_fixed)
        ax1.plot(x_d, d_vals, color=color, linewidth=2, label=f'$p={p_val}$')

        x_w, w_vals = m.compute_cod_profile(d_star_fixed)
        ax2.plot(x_w, w_vals, color=color, linewidth=2, label=f'$p={p_val}$')

    ax1.set_xlabel('Position $x$ (mm)', fontsize=11)
    ax1.set_ylabel('Phase-field $d(x)$', fontsize=11)
    ax1.set_title(f'Effect of $p$ on $d(x)$ Profile ($d^*={d_star_fixed}$)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Position $x$ (mm)', fontsize=11)
    ax2.set_ylabel('COD $w(x)$ (mm)', fontsize=11)
    ax2.set_title(f'Effect of $p$ on $w(x)$ Profile ($d^*={d_star_fixed}$)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'supp_p_effect_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Supp] Saved: {path}')
    return path


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  Paper Figure Reproduction & Visualization Suite')
    print('  uPF-CZM 1D Analytical Solutions')
    print('=' * 60)
    print(f'  Output directory: {OUTPUT_DIR}/')
    print(f'  Material: E=30 GPa, ft=3 MPa, Gf=0.12 N/mm')
    print()

    # Paper figures
    print('[1/7] Figure 11 — Crack profiles...')
    plot_figure_11()

    print('[2/7] Figure 12 — Cohesive laws...')
    plot_figure_12()

    print('[3/7] Combined Figure 11+12...')
    plot_figure_combined()

    # Supplementary
    print('[4/7] Softening comparison...')
    plot_softening_comparison()

    print('[5/7] Energy evolution...')
    plot_energy_evolution()

    print('[6/7] p-effect on profiles...')
    plot_p_effect_profiles()

    # Animation (takes longest)
    print('[7/7] Crack evolution animation...')
    model_anim = MicroPF_CZM_1D(**MATERIAL_PARAMS, softening='linear', p=1.0)
    anim_path = create_crack_evolution_animation(model_anim, fps=12, duration=10)

    print()
    print('=' * 60)
    print(f'  ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/')
    print('=' * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f'    {f} ({size_kb:.1f} KB)')
    print('=' * 60)
