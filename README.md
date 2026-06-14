# FractureNet-Ω

**Open-Source Implementation of the Generalized Phase-Field Cohesive Zone Model (μPF-CZM)**

Reference: Jian-Ying Wu (2024) — *A generalized phase-field cohesive zone model (μPF-CZM) for fracture*

---

## Overview

FractureNet-Ω is a **five-layer** computational framework for physics-informed fracture AI:

```
Layer 1: 1D Analytical Solver     →  Exact crack profiles, F*-u* curves
Layer 2: 2D/3D Finite Element     →  SENB benchmarks, Paraview VTK, HPC
Layer 3: Symbolic Regression       →  Automatic physics law discovery (R²=0.99999)
Layer 4: AI Parameter Identifier   →  CNN extracts material/crack params from fields
Layer 5: Unified Physics Model     →  Symbolic laws + Domain adaptation + Validation
```

---

## Project Structure

```
FractureNet-Ω/
├── src/
│   ├── micro_pf_czm_1d.py    # 1D μPF-CZM analytical solver (core)
│   ├── fem_2d.py              # 2D staggered FEM solver (paper Appendix D)
│   ├── mesh_utils.py          # SENB mesh generation + initial damage
│   └── vtk_export.py          # Paraview VTK/PVD export
├── generate_data.py           # Batch dataset generation (864 params)
├── discover_formula.py        # Symbolic regression (PySR + brute-force)
├── reproduce_paper_figures.py # Reproduce paper Fig.11 & Fig.12 + animation
├── Fu_curve_analysis.py       # F*-u* structural response curves
├── run_examples_suite.py      # Multi-case SENB benchmark suite
├── run_senb_benchmark.py      # Single SENB simulation
├── run_all.py                 # Full pipeline orchestrator
├── visualize.py               # Dataset visualization
├── paper/
│   └── main.tex               # LaTeX paper (CMAME format)
├── paper_figures/             # All generated figures + MP4 animation
└── fracture_data_extended.csv # Generated dataset (864 rows × 25 columns)
```

---

## Quick Start

### Installation

```bash
pip install numpy scipy matplotlib pandas scikit-fem
# Optional:
pip install pysr         # symbolic regression (requires Julia)
pip install numba        # JIT acceleration
```

### 1D Analytical Solver (Layer 1)

```python
from src.micro_pf_czm_1d import MicroPF_CZM_1D

model = MicroPF_CZM_1D(E=30000, ft=3.0, Gf=0.12, softening='linear', p=1.0, b=2.0)

# Crack band half-width
D = model.crack_half_width(d_star=0.8)

# Traction-separation law
w, sigma = model.traction_separation_law()

# Peak load with p- and softening-dependent size effect
Pmax = model.peak_load(L_char=100, width=100, height=100)

# Exact phase-field profile d(x)
x, d = model.compute_d_profile(d_star=0.8)

# COD profile w(x)
x, w = model.compute_cod_profile(d_star=0.8)
```

### 2D FEM Solver (Layer 2)

```python
from src.fem_2d import MuPFCZMMaterial, MuPFCZM2DSolver, find_senb_dofs
from src.mesh_utils import generate_senb_mesh_tri, set_notch_initial_damage

# Generate SENB mesh
mesh, notch_facets, beam_params = generate_senb_mesh_tri(D=50.0, h_el=2.0)

# Create material and solver
mat = MuPFCZMMaterial(E=30000, nu=0.2, Gf=0.12, ft=3.0, softening='linear', p=1.5, b=5.0)
solver = MuPFCZM2DSolver(mat, mesh)

# Set initial damage on notch
d_init = np.zeros(solver.ndof_d)
set_notch_initial_damage(mesh, notch_facets, d_init, b=5.0, smooth=True)
solver.set_initial_damage(d_init)

# Solve one load step
u_new, d_new, F, converged = solver.solve_step(
    -0.01, loading_dofs, support_dofs, n_stagger=10, tol=1e-4)
```

### Run Standard Benchmarks

```bash
# Reproduce paper figures + animation
python reproduce_paper_figures.py

# F*-u* structural response
python Fu_curve_analysis.py

# Symbolic regression discovery
python discover_formula.py --input fracture_data_extended.csv --method bruteforce

# Full pipeline
python run_all.py
```

---

## Accuracy Assessment

| Layer | Metric | Value | Method |
|-------|--------|:-----:|--------|
| **1D Analytical** | Gf conservation | **0.000000% error** | Exact paper Eq.(3.4) |
| **F*-u* Curves** | $R^2$ (ana vs num) | **1.00000000** | Derived $\bar{u}=\bar{F}+\lambda(1-\bar{F})$ |
| **Symbolic Regression** | $R^2$ | **0.999990** | Recovered $\sigma_N = B f_t/\sqrt{1+\beta_{\text{eff}}}$ |
| **2D FEM (framework)** | Formula match | **Exact** | Paper Appendix D, Eqs.(5.5)-(5.6) |

### Discovered Formula

The symbolic regression automatically recovered the extended size-effect law:

$$\sigma_N = 0.5055 \cdot f_t \Big/ \sqrt{1 + 1.0212 \cdot \beta \cdot \frac{2p+1}{3} \Big/ S_{\text{soft}}}$$

with $R^2 = 0.999990$ and RMSE = 0.0015 MPa.

### Implemented Characteristic Functions (Paper Appendix D)

| Function | Formula | Status |
|----------|---------|:------:|
| Geometric | $\alpha(d) = 2d - d^2$ | ✅ Exact |
| Degradation | $\omega(d) = 1/(1+\phi(d))$ | ✅ Exact |
| Auxiliary | $\phi(d) = a_0\sqrt{\alpha}/(1-d)^{p+1} \cdot \Xi(d)$ | ✅ Exact |
| Dissipation | $\varpi'(d) = -\omega^2(d)\mu'(d)$ | ✅ Exact |
| Mu-function | $\mu(d) = a_0\alpha(d)/(1-d)^{2p}$ | ✅ Exact |

### Softening Curves Supported

| Type | $\Xi(d)$ | Application |
|------|----------|-------------|
| Linear | $s$ | Reference |
| Exponential | $\frac{1}{2}\text{arctanh}(s)$ | Ductile materials |
| Cornelissen | 6th-order polynomial | Concrete |
| PPR (Park-Paulino-Roesler) | Power-law | Adhesives |

---

## Computational Complexity

### Scaling Laws

```
n_nodes  ∝ 4.5 × (D/h)²           Grid nodes
t_step   ∝ n_nodes^1.4            Sparse direct solver dominant
t_full   = t_step × N_steps × N_stagger × 1.5 (assembly)
```

### Empirical Timing (scipy single-thread, laptop CPU)

| Test | Nodes | Elements | Time/Step |
|------|------:|------:|------:|
| 12mm beam, h=2.0mm | 196 | 324 | 1.0 s |
| 25mm beam, h=2.0mm | 741 | 1,344 | 6.6 s |
| 30mm beam, h=1.0mm | 4,216 | 8,100 | 50.0 s |

### Full Benchmark Projections (20 steps × 8 stagger iterations)

| Configuration | D | h | Nodes | Laptop (scipy) | Workstation (MKL 8c) | HPC (32c) | GPU (CUDA) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Demo** | 15mm | 1.5mm | 200 | **2.7 min** | 13 s | 3 s | 1 s |
| Coarse | 25mm | 1.0mm | 1,200 | 34 min | 2.7 min | 40 s | 10 s |
| **Medium** | 50mm | 1.0mm | 11,250 | 12.9 h | **1.0 h** | 15 min | 3.9 min |
| Fine | 100mm | 1.0mm | 45,000 | 3.7 d | 7.2 h | 1.8 h | **27 min** |
| Production | 200mm | 1.0mm | 180,000 | 26 d | 2.1 d | 12.5 h | 3.1 h |
| High-fidelity | 200mm | 0.5mm | 720,000 | 181 d | 14.5 d | 3.6 d | 21.8 h |

### Hardware Speedup Factors (relative to scipy single-thread)

| Platform | Speedup | Recommendation |
|----------|:-------:|----------------|
| scipy (laptop, 1 core) | 1× | Development / demo |
| Desktop i7/i9 | 1.5-2× | Small tests |
| FEniCS (C++ backend) | 20× | Python production |
| Workstation (MKL, 8 cores) | 12× | Medium benchmarks |
| HPC node (32 cores) | 50× | Full size effect |
| GPU (CUDA sparse) | 200× | Large-scale production |

### Strategy

- **Parameter studies**: Use 1D analytical solver (864 combinations in 1.1 seconds)
- **Crack visualization**: Coarse 2D FEM with VTK output
- **Paper validation**: Medium mesh on workstation (~1 hour)
- **Production**: HPC or GPU-accelerated solver

---

## Visualizations

### Paper Figures Reproduced

| Figure | Content | Output |
|--------|---------|--------|
| Fig.11(a) | Crack phase-field profiles $d(x)$ | `fig11_crack_profiles.png` |
| Fig.11(b) | COD profiles $w(x)$ | `fig11_crack_profiles.png` |
| Fig.12(a) | Traction-separation laws | `fig12_cohesive_laws.png` |
| Fig.12(b) | $w(0)$ vs $d^*$ relation | `fig12_cohesive_laws.png` |
| Combined | 4-panel publication figure | `fig11_12_combined.png` |
| Animation | Crack evolution MP4 | `crack_evolution.mp4` |

### F*-u* Analysis

| Figure | Content |
|--------|---------|
| `Fu_size_effect.png` | Load-displacement for 7 structural sizes |
| `Fu_p_effect.png` | Effect of traction order $p$ |
| `Fu_softening_effect.png` | Effect of softening type |
| `Fu_paper_validation.png` | 4-panel: size effect + peak load + Gf + ductility |
| `Fu_comprehensive_summary.png` | 5-panel publication summary |

### Paraview 3D Visualization

```bash
# Run examples with VTK output
python run_examples_suite.py

# In Paraview:
# File → Open → examples_output/vtk/<case>/step.pvd
# Click ▶ to animate crack propagation
# Color by: phase_field
```

---

## Key Physics Implemented

### 1D Bar under Tension

- Crack phase-field profile: $x(d;d^*) = b\int_d^{d^*} [\sqrt{\alpha(\theta)}(1-(1-\theta)^{2p}/\eta_0)]^{-1} d\theta$
- Center COD: $w(0;d^*) = w_c \cdot \omega^{-1}((1-d^*)^{2p})$
- Structural response: $F^*(d^*) = f_t(1-d^*)^{2p}A$, $u^*(d^*) = \sigma\cdot 2L/E + w(0)$
- Exact post-peak relation ($p=1$, linear): $\bar{u} = \bar{F} + \lambda(1-\bar{F})$

### 2D FEM (Paper Eqs. 5.5-5.6)

- Displacement stiffness: $\mathbf{K}_{uu} = \int \mathbf{B}^T \omega(d)\mathbb{C} \mathbf{B} \,dV$
- Phase-field tangent: $\mathbf{K}_{dd} = \int [\bar{\mathbf{N}}^T(\varpi''\bar{Y} + \alpha''\frac{G_f}{c_\alpha b})\bar{\mathbf{N}} + \frac{2b}{c_\alpha}G_f\bar{\mathbf{B}}^T\bar{\mathbf{B}}] dV$
- Staggered alternate minimization with irreversibility $d_{n+1} \geq d_n$

### Extended Size Effect Law

$$\sigma_N = B f_t \Big/ \sqrt{1 + \beta \cdot \frac{2p+1}{3} \Big/ S_{\text{soft}} \Big/ f_b}$$

where $\beta = l_{ch}/L_{char}$, $S_{\text{soft}}$ = softening shape factor, $f_b = 1 + D(d^*)/l_{ch}$.

---

## Dependencies

```
numpy>=1.21        scipy>=1.7         matplotlib>=3.4
pandas>=1.3        scikit-fem>=9.0    pysr>=0.7 (optional)
```

---

## Reference

Wu, J.Y. (2024). *A generalized phase-field cohesive zone model (μPF-CZM) for fracture.* arXiv:2408.00015v1.

---

## License

MIT License. Free for academic and commercial use.

---

## HPC & 3D Usage Guide

### Directory Structure

```
FractureNet-Ω/
├── src/                          # Core solvers
│   ├── micro_pf_czm_1d.py        # 1D analytical solver
│   ├── fem_2d.py                 # 2D staggered FEM (paper Appendix D)
│   ├── mesh_utils.py             # SENB mesh + initial damage
│   └── vtk_export.py             # Paraview VTK/PVD export
├── hpc/                          # HPC-ready modules
│   ├── fem3d_solver.py           # 3D uPF-CZM FEM solver
│   ├── mesostructure/
│   │   └── concrete_generator.py # Concrete aggregate+ITZ generator
│   ├── dataset/
│   │   └── pipeline.py           # E2E dataset pipeline + SLURM gen
│   └── gnn/
│       └── graph_builder.py      # GNN graph data constructor
├── paper_figures/                # All generated figures
├── patent/                       # Patent document (.docx)
├── paper/                        # LaTeX paper (CMAME format)
├── run_all.py                    # Full pipeline orchestrator
├── run_examples_suite.py         # Multi-case benchmark suite
├── requirements.txt
└── 改进的吴氏模型.pdf            # Reference paper (Wu 2024)
```

### 1. Quick Local Test (All Modules)

```bash
# Verify all modules work (30s)
python -c "
import sys; sys.path.insert(0,'src')
from hpc.fem3d_solver import generate_cube_with_notch, Solver3D
from fem_2d import MuPFCZMMaterial
mesh = generate_cube_with_notch(8,8,8,2.0,3.0)
mat = MuPFCZMMaterial(E=5000,nu=0.2,Gf=0.12,ft=3.0)
s = Solver3D(mat, mesh); print(f'3D: {mesh.n_nodes} nodes, K0 nnz={s.assemble_undamaged_stiffness().nnz}')

from hpc.mesostructure.concrete_generator import ConcreteSpec, generate_2d_concrete
spec = ConcreteSpec(Lx=30,Ly=30,voxel_size=1.0,vol_frac_agg=0.4)
v,_ = generate_2d_concrete(spec); print(f'Concrete: {v.shape}')

from hpc.gnn.graph_builder import build_graph_from_fem
import numpy as np
g = build_graph_from_fem(np.random.rand(2,50)*10, np.array([[0,1,2]]).T,
    np.random.rand(50), np.random.rand(100)*0.01, k_nn=4)
print(f'GNN: {g.edge_index.shape[1]} edges')
print('ALL OK')
"
```

### 2. Generate Concrete Mesostructure

```bash
# 2D concrete with 45% aggregate volume
python hpc/mesostructure/concrete_generator.py \
    --dim 2 --size 100 --vol_frac 0.45 --voxel_size 0.5 \
    --output concrete_2d.npz --plot

# 3D concrete (for HPC)
python hpc/mesostructure/concrete_generator.py \
    --dim 3 --size 50 --vol_frac 0.40 --voxel_size 0.5 \
    --output concrete_3d.npz
```

### 3. Run Full Dataset Pipeline (Local)

```bash
# Quick demo: 1D + 2D + mesostructure + GNN graphs
python hpc/dataset/pipeline.py --quick --output_dir dataset_demo

# Full production (may take hours on 2D FEM)
python hpc/dataset/pipeline.py --output_dir dataset_full
```

### 4. Deploy on HPC Cluster

```bash
# Step 1: Copy code to cluster
rsync -avz FractureNet-Ω/ user@cluster:/path/to/

# Step 2: Install dependencies
ssh user@cluster "pip install -r /path/to/requirements.txt scikit-fem"

# Step 3: Generate SLURM script + run
python hpc/dataset/pipeline.py --output_dir /scratch/$USER/dataset
sbatch hpc/jobs/pipeline_slurm.sh

# Step 4: Monitor
squeue -u $USER
ls /scratch/$USER/dataset/
cat /scratch/$USER/dataset/manifest.json
```

### 5. Build GNN Dataset from FEM Results

```python
from hpc.gnn.graph_builder import build_temporal_graph_sequence, save_graph_dataset
import numpy as np

# Load your FEM results (d_history, u_history, mesh)
# ...

graphs = build_temporal_graph_sequence(mesh_nodes, mesh_tets,
                                        d_history, u_history,
                                        material_field=material_props)
save_graph_dataset(graphs, 'gnn_output/', prefix='sample')
# Output: gnn_output/sample_0000.npz ... sample_XXXX.npz + metadata.json
```

### 6. Visualize Results in Paraview

```
1. Open Paraview
2. File → Open → <case_dir>/vtk/step.pvd
3. Click ▶ to animate crack propagation
4. Color by: phase_field
5. Add filter: Contour (d=0.9) to show crack surface
```

### 7. Run Individual Benchmarks

| Script | Purpose | Time |
|--------|---------|:----:|
| `python reproduce_paper_figures.py` | Fig.11+12 + MP4 animation | <1 min |
| `python Fu_curve_analysis.py` | F*-u* size effect curves | <1 min |
| `python discover_formula.py` | Symbolic regression discovery | <1 min |
| `python run_examples_suite.py` | 8-case SENB + VTK export | 10-30 min |
| `python run_senb_benchmark.py` | Single SENB simulation | 2-10 min |

### 8. Patent & Paper

| File | Content |
|------|---------|
| `patent/CN_PATENT_FractureNet_Omega.docx` | Complete invention patent (Word) |
| `patent/PATENT_TIMESTAMP.txt` | Timestamp proof (UTC + hash) |
| `paper/main.tex` | LaTeX paper (CMAME format) |
| `paper_figures/` | All paper figures (PNG + MP4) |

### 9. Key Parameters Reference

| Parameter | Symbol | Typical Range | Notes |
|-----------|--------|:------:|-------|
| Young's modulus | E | 5000-50000 MPa | Lower for meso, full for macro |
| Tensile strength | ft | 1.5-6.0 MPa | ITZ weakest at 1.5 MPa |
| Fracture energy | Gf | 0.03-0.20 N/mm | ITZ lowest at 0.03 N/mm |
| Length scale | b | 1-20 mm | h <= b/5 for accuracy |
| Traction order | p | 1.0-3.0 | p=1 linear, p>1 steeper |
| Aggregate vol frac | V_agg | 0.35-0.50 | Typical concrete 40-45% |
| ITZ thickness | t_ITZ | 0.2-1.0 mm | Typically 20-50 um, scaled for mesh |

### 10. Citation

If you use this code, please cite:

```bibtex
@software{FractureNet-Omega,
  title = {FractureNet-$\Omega$: Open-Source Implementation of $\mu$PF-CZM},
  year = {2026},
  note = {Three-layer computational framework for phase-field fracture}
}

@article{Wu2024muPF,
  title = {A generalized phase-field cohesive zone model ($\mu$PF-CZM) for fracture},
  author = {Wu, Jian-Ying},
  journal = {arXiv:2408.00015v1},
  year = {2024}
}
```
