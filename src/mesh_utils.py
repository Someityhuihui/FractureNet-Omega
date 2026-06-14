"""
SENB Mesh Generation Utilities
===============================
Generates structured quadrilateral meshes for the
Single-Edge-Notched Beam (SENB) three-point bending test.

Standard geometry (from RILEM TC 89-FMT / Elices et al.):
  - Beam depth: D
  - Span: S = 4D
  - Total length: L = 4.5D (with 0.25D overhang at each end)
  - Notch depth: a0 (typically 0.2D for size effect studies)
  - Notch width: w0 (typically ~0.01D, modeled as initial d=1)

The mesh uses graded refinement toward the notch tip to
resolve the fracture process zone.
"""

import numpy as np
from skfem import MeshTri, MeshQuad


def generate_senb_mesh_quad(D=100.0, a0_ratio=0.2, n_x=80, n_y=40,
                             refine_factor=3.0):
    """
    Generate a structured quadrilateral mesh for the SENB beam.

    The mesh is graded: element size decreases toward the notch tip
    (center-bottom of beam) to resolve the crack path.

    Parameters
    ----------
    D : float
        Beam depth (mm)
    a0_ratio : float
        Notch depth ratio a0/D, default 0.2
    n_x : int
        Number of elements along x-direction
    n_y : int
        Number of elements along y-direction
    refine_factor : float
        Mesh grading factor (>1 means finer near notch tip)

    Returns
    -------
    mesh : skfem.MeshQuad
        Quadrilateral mesh of the beam
    notch_nodes : ndarray
        Node indices lying on the notch boundary
    beam_params : dict
        Geometry parameters for post-processing
    """
    S = 4.0 * D           # span
    L = 4.5 * D           # total beam length
    a0 = a0_ratio * D     # notch depth
    w_notch = 0.01 * D    # notch width (modeled by initial damage)

    # Generate graded node positions
    # x-direction: uniform (could add grading near center)
    x_nodes = np.linspace(0, L, n_x + 1)

    # y-direction: graded toward top (crack propagates upward)
    # Finer elements near y = a0 (notch tip), coarser near y = D (top)
    y_start = 0.0      # bottom of beam
    y_notch = a0       # notch tip
    y_top = D          # top of beam

    # Create graded y-coordinates
    n_y_below = int(n_y * a0_ratio)      # elements below notch tip
    n_y_above = n_y - n_y_below          # elements above notch tip

    # Below notch: fine near tip
    y_below = y_notch * (1.0 - np.cos(np.linspace(0, np.pi/2, n_y_below + 1))) ** refine_factor
    y_below = y_below / y_below.max() * y_notch

    # Above notch: fine near tip, coarser toward top
    y_above_raw = np.linspace(0, 1, n_y_above + 1)
    y_above = y_notch + (D - y_notch) * (y_above_raw) ** (1.0 / refine_factor)

    # Combine
    y_nodes = np.concatenate([y_below[:-1], y_above])

    # Create MeshQuad
    mesh = MeshQuad()
    # We'll build the mesh manually using p and t arrays
    n_nodes_x = len(x_nodes)
    n_nodes_y = len(y_nodes)

    # Create node coordinates
    p = np.zeros((2, n_nodes_x * n_nodes_y))
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            node_id = j * n_nodes_x + i
            p[0, node_id] = x_nodes[i]
            p[1, node_id] = y_nodes[j]

    # Create quad connectivity
    n_elem_x = n_nodes_x - 1
    n_elem_y = n_nodes_y - 1
    n_elem = n_elem_x * n_elem_y
    t = np.zeros((4, n_elem), dtype=int)

    for j in range(n_elem_y):
        for i in range(n_elem_x):
            elem_id = j * n_elem_x + i
            # Q1 connectivity (counter-clockwise)
            n0 = j * n_nodes_x + i
            n1 = n0 + 1
            n2 = n0 + n_nodes_x + 1
            n3 = n0 + n_nodes_x
            t[:, elem_id] = [n0, n1, n2, n3]

    # Build the mesh
    mesh = MeshQuad(p, t)

    # Identify notch boundary nodes (for initial damage)
    center_x = L / 2.0
    notch_nodes = []
    for node_id in range(p.shape[1]):
        if (abs(p[0, node_id] - center_x) < w_notch and
                p[1, node_id] <= a0 * 1.01):
            notch_nodes.append(node_id)

    beam_params = {
        'D': D, 'S': S, 'L': L, 'a0': a0,
        'w_notch': w_notch, 'center_x': center_x,
        'n_nodes_x': n_nodes_x, 'n_nodes_y': n_nodes_y,
    }

    return mesh, np.array(notch_nodes), beam_params


def generate_senb_mesh_tri(D=100.0, a0_ratio=0.2, h_el=3.0,
                            refine_radius=15.0, h_refine=1.0):
    """
    Generate a triangular mesh for the SENB beam using MeshTri.

    Uses uniform triangulation with optional refinement near notch tip.
    Simpler than the graded quad mesh.

    Parameters
    ----------
    D : float
        Beam depth (mm)
    a0_ratio : float
        Notch depth ratio a0/D
    h_el : float
        Target element size (mm) in the bulk
    refine_radius : float
        Radius around notch tip for refinement
    h_refine : float
        Target element size (mm) near notch tip

    Returns
    -------
    mesh : skfem.MeshTri
        Triangular mesh
    notch_facets : ndarray
        Facet indices on the notch boundary
    beam_params : dict
        Geometry parameters
    """
    S = 4.0 * D
    L = 4.5 * D
    a0 = a0_ratio * D
    center_x = L / 2.0

    # Use int for element counts
    n_x = int(L / h_el) + 1
    n_y = int(D / h_el) + 1

    # Generate structured triangular mesh
    # Create rectangular mesh and then mark notch
    x = np.linspace(0, L, n_x)
    y = np.linspace(0, D, n_y)
    X, Y = np.meshgrid(x, y)

    p = np.vstack([X.ravel(), Y.ravel()])

    # Create two triangles per rectangle
    n_elem_x = n_x - 1
    n_elem_y = n_y - 1
    n_tri = 2 * n_elem_x * n_elem_y
    t = np.zeros((3, n_tri), dtype=int)

    tri_id = 0
    for j in range(n_elem_y):
        for i in range(n_elem_x):
            n00 = j * n_x + i
            n10 = n00 + 1
            n01 = n00 + n_x
            n11 = n01 + 1
            # Lower-left triangle
            t[:, tri_id] = [n00, n10, n11]
            tri_id += 1
            # Upper-right triangle
            t[:, tri_id] = [n00, n11, n01]
            tri_id += 1

    mesh = MeshTri(p, t)

    # Identify notch facets (for d=1 initial condition)
    # Facets on x = center_x, y in [0, a0]
    notch_facets = []
    for fid in range(mesh.facets.shape[1]):
        facet_nodes = mesh.facets[:, fid]
        mid_x = np.mean(p[0, facet_nodes])
        mid_y = np.mean(p[1, facet_nodes])
        if abs(mid_x - center_x) < h_el / 2 and mid_y < a0 + h_el / 2:
            notch_facets.append(fid)

    beam_params = {
        'D': D, 'S': S, 'L': L, 'a0': a0,
        'center_x': center_x, 'h_el': h_el,
    }

    return mesh, np.array(notch_facets), beam_params


def set_notch_initial_damage(mesh, notch_nodes_or_facets, dof_array,
                              b=5.0, smooth=True):
    """
    Set initial damage on the notch with a smoothed transition zone.

    Without smoothing: d=1 exactly on notch nodes → stress singularity.
    With smoothing: d = exp(-dist/b) profile over ~b width from notch.

    Parameters
    ----------
    mesh : skfem.Mesh
    notch_nodes_or_facets : ndarray
        Node or facet indices on the notch boundary
    dof_array : ndarray
        Phase-field DOF array (modified in place)
    b : float
        Phase-field length scale for smoothing width
    smooth : bool
        If True, apply smoothed profile; if False, set d=1 on notch only
    """
    if len(notch_nodes_or_facets) == 0:
        return

    # Collect notch node indices
    if notch_nodes_or_facets.ndim == 1:
        notch_nodes = list(notch_nodes_or_facets)
    else:
        notch_nodes = list(set(
            mesh.facets[:, fid].flatten()
            for fid in range(notch_nodes_or_facets.shape[1])
            if fid < len(notch_nodes_or_facets)
        ))
        # Flatten the list of arrays
        flat_nodes = []
        for fid in range(notch_nodes_or_facets.shape[1]):
            for nid in mesh.facets[:, fid]:
                flat_nodes.append(int(nid))
        notch_nodes = list(set(flat_nodes))

    if not smooth or b <= 0:
        for nid in notch_nodes:
            dof_array[nid] = 1.0
        return

    # Smoothed profile: d = exp(-dist^2 / (2*b^2))
    p = mesh.p
    # Compute distances from notch for all nodes
    for i in range(p.shape[1]):
        if i in notch_nodes:
            dof_array[i] = 1.0
            continue
        # Min distance to any notch node
        min_dist = min(np.sqrt((p[0,i] - p[0,j])**2 + (p[1,i] - p[1,j])**2)
                       for j in notch_nodes[:min(10, len(notch_nodes))])
        # Gaussian profile
        dof_array[i] = np.exp(-0.5 * (min_dist / b)**2)


# ================================================================
# Test
# ================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    D_test = 100.0

    # Test quad mesh
    mesh_q, notch_n, bp = generate_senb_mesh_quad(D=D_test, n_x=40, n_y=20)
    print(f"Quad mesh: {mesh_q.p.shape[1]} nodes, {mesh_q.t.shape[1]} elements")
    print(f"  Notch nodes: {len(notch_n)}")
    print(f"  Beam: {bp['L']:.0f} x {bp['D']:.0f} mm, notch={bp['a0']:.0f} mm")

    # Quick plot
    fig, ax = plt.subplots(figsize=(10, 3))
    mesh_q.draw(ax=ax, boundaries_only=True)
    ax.scatter(mesh_q.p[0, notch_n], mesh_q.p[1, notch_n],
               c='red', s=5, label='Notch')
    ax.set_title(f'Quad Mesh: {bp["L"]:.0f}x{bp["D"]:.0f} mm SENB')
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig('mesh_preview_quad.png', dpi=150)
    print("  Saved: mesh_preview_quad.png")

    # Test tri mesh
    mesh_t, notch_f, bp2 = generate_senb_mesh_tri(D=D_test, h_el=4.0)
    print(f"\nTri mesh: {mesh_t.p.shape[1]} nodes, {mesh_t.t.shape[1]} elements")
    print(f"  Notch facets: {len(notch_f)}")

    fig, ax = plt.subplots(figsize=(10, 3))
    mesh_t.draw(ax=ax, boundaries_only=True)
    notch_nodes_set = set()
    for fid in notch_f:
        for nid in mesh_t.facets[:, fid]:
            notch_nodes_set.add(nid)
    notch_nodes_arr = np.array(list(notch_nodes_set))
    ax.scatter(mesh_t.p[0, notch_nodes_arr], mesh_t.p[1, notch_nodes_arr],
               c='red', s=5, label='Notch')
    ax.set_title(f'Tri Mesh: {bp2["L"]:.0f}x{bp2["D"]:.0f} mm SENB')
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig('mesh_preview_tri.png', dpi=150)
    print("  Saved: mesh_preview_tri.png")
    plt.close('all')
