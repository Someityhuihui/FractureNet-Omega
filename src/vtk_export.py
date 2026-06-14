"""
VTK Export for Paraview Visualization
======================================
Exports uPF-CZM 2D FEM results to VTK format for Paraview.
Supports triangular meshes with displacement and phase-field data.
"""

import numpy as np
import os


def export_vtk(mesh, u_field, d_field, filename, beam_params=None):
    """
    Export 2D FEM results to VTK unstructured grid format.

    Parameters
    ----------
    mesh : skfem.MeshTri
    u_field : ndarray — displacement DOFs (length 2*n_nodes)
    d_field : ndarray — phase-field DOFs (length n_nodes)
    filename : str — output .vtu file path
    beam_params : dict, optional
    """
    p = mesh.p  # (2, n_nodes)
    t = mesh.t  # (3, n_elements)
    n_nodes = p.shape[1]
    n_elems = t.shape[1]

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write('<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{n_nodes}" '
                f'NumberOfCells="{n_elems}">\n')

        # --- Points ---
        f.write('<Points>\n')
        f.write('<DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n')
        for i in range(n_nodes):
            f.write(f'{p[0,i]:.6f} {p[1,i]:.6f} 0.0\n')
        f.write('</DataArray>\n')
        f.write('</Points>\n')

        # --- Cells ---
        f.write('<Cells>\n')
        # Connectivity
        f.write('<DataArray type="Int32" Name="connectivity" '
                'format="ascii">\n')
        for e in range(n_elems):
            f.write(f'{t[0,e]} {t[1,e]} {t[2,e]}\n')
        f.write('</DataArray>\n')
        # Offsets
        f.write('<DataArray type="Int32" Name="offsets" '
                'format="ascii">\n')
        for e in range(1, n_elems + 1):
            f.write(f'{3*e}\n')
        f.write('</DataArray>\n')
        # Types (5 = VTK_TRIANGLE)
        f.write('<DataArray type="UInt8" Name="types" '
                'format="ascii">\n')
        for e in range(n_elems):
            f.write('5\n')
        f.write('</DataArray>\n')
        f.write('</Cells>\n')

        # --- Point Data ---
        f.write('<PointData>\n')

        # Phase-field
        f.write('<DataArray type="Float64" Name="phase_field" '
                'format="ascii">\n')
        for i in range(n_nodes):
            f.write(f'{d_field[i]:.8f}\n')
        f.write('</DataArray>\n')

        # Displacement magnitude
        f.write('<DataArray type="Float64" Name="displacement" '
                'NumberOfComponents="3" format="ascii">\n')
        for i in range(n_nodes):
            ux = u_field[2*i] if 2*i < len(u_field) else 0.0
            uy = u_field[2*i+1] if 2*i+1 < len(u_field) else 0.0
            f.write(f'{ux:.8f} {uy:.8f} 0.0\n')
        f.write('</DataArray>\n')

        f.write('</PointData>\n')

        f.write('</Piece>\n')
        f.write('</UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

    size_kb = os.path.getsize(filename) / 1024
    print(f'  VTK exported: {filename} ({size_kb:.1f} KB)')


def export_vtk_sequence(solver, mesh, u_history_list, d_history_list,
                        output_dir='vtk_output', prefix='step'):
    """
    Export a sequence of VTK files for animation in Paraview.

    Parameters
    ----------
    solver : MuPFCZM2DSolver
    mesh : skfem.MeshTri
    u_history_list : list of ndarray — displacement at each step
    d_history_list : list of ndarray — phase-field at each step
    output_dir : str
    prefix : str — filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (u, d) in enumerate(zip(u_history_list, d_history_list)):
        fname = os.path.join(output_dir, f'{prefix}_{i+1:04d}.vtu')
        export_vtk(mesh, u, d, fname)

    # Write Paraview collection file (.pvd)
    pvd_path = os.path.join(output_dir, f'{prefix}.pvd')
    with open(pvd_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('<Collection>\n')
        n_steps = len(u_history_list)
        for i in range(n_steps):
            fname = f'{prefix}_{i+1:04d}.vtu'
            f.write(f'<DataSet timestep="{i}" file="{fname}"/>\n')
        f.write('</Collection>\n')
        f.write('</VTKFile>\n')

    print(f'  PVD collection: {pvd_path} ({n_steps} timesteps)')


# ====================================================================
# Quick test
# ====================================================================
if __name__ == '__main__':
    from mesh_utils import generate_senb_mesh_tri
    import numpy as np

    mesh, _, bp = generate_senb_mesh_tri(D=20.0, h_el=4.0)
    u_test = np.zeros(2 * mesh.p.shape[1])
    d_test = np.random.rand(mesh.p.shape[1])

    os.makedirs('vtk_output', exist_ok=True)
    export_vtk(mesh, u_test, d_test, 'vtk_output/test.vtu')
    print('VTK export works! Open vtk_output/test.vtu in Paraview.')
