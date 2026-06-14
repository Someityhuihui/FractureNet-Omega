"""
Graph Neural Network Data Interface
=====================================
Converts uPF-CZM FEM results into graph-structured data for GNN training.

Graph structure:
  - Nodes: FEM mesh nodes with features [x, y, z, d, u_x, u_y, u_z, E, nu, ft, Gf]
  - Edges: mesh connectivity + k-NN edges for long-range crack interaction
  - Node targets: d(t+1) — next timestep phase-field prediction
  - Edge features: relative displacement, distance

The graph can be consumed by:
  - GCN (Graph Convolutional Network)
  - GAT (Graph Attention Network)
  - MeshGraphNet (Pfaff et al., 2021)
  - GNN with physics-informed loss (energy conservation, irreversibility)

Usage:
  python graph_builder.py --input vtk_sequence/ --output dataset/
"""

import numpy as np
from scipy.spatial import KDTree
import os, sys, json, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class GraphData:
    """Single-timestep graph."""
    node_features: np.ndarray     # (n_nodes, n_features)
    edge_index: np.ndarray         # (2, n_edges)
    edge_features: np.ndarray      # (n_edges, n_edge_features)
    node_targets: np.ndarray       # (n_nodes,) — d at next step
    global_features: np.ndarray    # (n_global,) — load, step, E_eff, etc.


def build_graph_from_fem(mesh_nodes, mesh_tets, d_field, u_field,
                         material_field=None, next_d_field=None,
                         k_nn=8, include_material=True):
    """
    Build a graph from 2D/3D FEM results.

    Parameters
    ----------
    mesh_nodes : ndarray (dim, n_nodes)
    mesh_tets : ndarray (n_verts_per_elem, n_elems)
    d_field : ndarray (n_nodes,) — current phase-field
    u_field : ndarray (dim*n_nodes,) — current displacement
    material_field : ndarray (n_nodes, n_mat_props) or None
    next_d_field : ndarray (n_nodes,) — next timestep d (target)
    k_nn : int — number of nearest neighbors for long-range edges
    include_material : bool

    Returns
    -------
    GraphData
    """
    dim = mesh_nodes.shape[0]
    n_nodes = mesh_nodes.shape[1]

    # === Node features ===
    feats = [mesh_nodes.T]  # (n_nodes, dim) — positions
    feats.append(d_field.reshape(-1, 1))  # phase-field

    u_reshaped = u_field.reshape(dim, n_nodes).T if dim == 2 else \
        u_field.reshape(3, n_nodes).T
    feats.append(u_reshaped)  # displacement

    if material_field is not None:
        feats.append(material_field)

    node_features = np.hstack(feats)

    # === Edge index (mesh connectivity + k-NN) ===
    edges_set = set()
    # Mesh edges
    n_verts = mesh_tets.shape[0]
    for e in range(mesh_tets.shape[1]):
        elem_nodes = mesh_tets[:, e]
        for i in range(n_verts):
            for j in range(i+1, n_verts):
                ni, nj = int(elem_nodes[i]), int(elem_nodes[j])
                edges_set.add((min(ni, nj), max(ni, nj)))

    # k-NN edges (long-range)
    tree = KDTree(mesh_nodes.T)
    for i in range(n_nodes):
        dists, idxs = tree.query(mesh_nodes[:, i], k=min(k_nn+1, n_nodes))
        for j_idx, d in zip(idxs[1:], dists[1:]):
            if d < 3.0 * np.mean(dists[1:]):  # radius limit
                edges_set.add((min(i, j_idx), max(i, j_idx)))

    edge_list = sorted(edges_set)
    n_edges = len(edge_list)
    edge_index = np.array(edge_list).T  # (2, n_edges)

    # === Edge features ===
    edge_feats_list = []
    for (src, dst) in edge_list:
        pos_diff = mesh_nodes[:, dst] - mesh_nodes[:, src]
        dist = np.linalg.norm(pos_diff)
        d_diff = d_field[dst] - d_field[src]
        u_diff = u_field.reshape(dim, n_nodes)[:, dst] - \
            u_field.reshape(dim, n_nodes)[:, src]
        edge_feats_list.append(
            np.concatenate([pos_diff, [dist], [d_diff], u_diff]))
    edge_features = np.array(edge_feats_list)

    # === Targets ===
    if next_d_field is not None:
        targets = next_d_field
    else:
        targets = d_field.copy()  # identity for inference

    # === Global features ===
    d_max = d_field.max()
    d_mean = d_field.mean()
    u_norm = np.linalg.norm(u_field)
    global_features = np.array([d_max, d_mean, u_norm])

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_targets=targets,
        global_features=global_features,
    )


def build_temporal_graph_sequence(mesh_nodes, mesh_tets, d_history, u_history,
                                  material_field=None, k_nn=8):
    """
    Build a sequence of graphs for temporal GNN training.

    Each graph includes (t, t+1) as (input, target).

    Returns
    -------
    List[GraphData] — one per adjacent timestep pair
    """
    graphs = []
    n_steps = len(d_history)

    for t in range(n_steps - 1):
        g = build_graph_from_fem(
            mesh_nodes, mesh_tets,
            d_history[t], u_history[t],
            material_field=material_field,
            next_d_field=d_history[t+1],
            k_nn=k_nn,
        )
        # Add temporal features
        g.node_features = np.hstack([
            g.node_features,
            np.full((g.node_features.shape[0], 1), t / max(1, n_steps-1))
        ])
        graphs.append(g)

    return graphs


def save_graph_dataset(graphs, output_dir, prefix='sample'):
    """Save graph dataset to compressed numpy format."""
    os.makedirs(output_dir, exist_ok=True)

    for i, g in enumerate(graphs):
        np.savez_compressed(
            os.path.join(output_dir, f'{prefix}_{i:04d}.npz'),
            node_features=g.node_features,
            edge_index=g.edge_index,
            edge_features=g.edge_features,
            node_targets=g.node_targets,
            global_features=g.global_features,
        )

    # Metadata
    meta = {
        'n_samples': len(graphs),
        'n_nodes': int(graphs[0].node_features.shape[0]),
        'n_node_features': int(graphs[0].node_features.shape[1]),
        'n_edges': int(graphs[0].edge_index.shape[1]),
        'n_edge_features': int(graphs[0].edge_features.shape[1]),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {len(graphs)} graphs to {output_dir}/")
    print(f"  Metadata: {json.dumps(meta)}")


def physics_informed_loss(pred_d, true_d, graph, material):
    """
    Physics-informed loss components for GNN training.

    Loss = MSE(pred, true)
         + alpha * Irreversibility violation (pred_d < prev_d)
         + beta  * Energy balance violation
         + gamma * Bounds violation (d < 0 or d > 1)

    Parameters
    ----------
    pred_d : ndarray — predicted d
    true_d : ndarray — ground truth d
    graph : GraphData
    material : material parameters dict

    Returns
    -------
    dict of loss components
    """
    mse = np.mean((pred_d - true_d)**2)

    # Irreversibility: penalize d decreasing
    prev_d = graph.node_features[:, 2]  # assuming d is feature index 2
    irrev_violation = np.mean(np.maximum(0, prev_d - pred_d)**2)

    # Bounds: penalize d outside [0, 1]
    bound_violation = (np.mean(np.maximum(0, -pred_d)**2) +
                       np.mean(np.maximum(0, pred_d - 1.0)**2))

    return {
        'mse': mse,
        'irreversibility': irrev_violation,
        'bounds': bound_violation,
        'total': mse + 0.1*irrev_violation + 0.5*bound_violation,
    }


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  GNN Graph Builder — Test")
    print("=" * 55)

    # Dummy 2D test mesh
    n_nodes = 100
    mesh_nodes = np.random.rand(2, n_nodes) * 10
    mesh_tets = np.array([
        [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
        [5, 6, 7], [6, 7, 8], [7, 8, 9], [0, 1, 2],
    ]).T[:, :5]  # dummy connectivity

    d_field = np.random.rand(n_nodes)
    u_field = np.random.rand(2 * n_nodes) * 0.01
    d_next = d_field + np.random.rand(n_nodes) * 0.001

    g = build_graph_from_fem(mesh_nodes, mesh_tets, d_field, u_field,
                             next_d_field=d_next, k_nn=4)

    print(f"  Nodes: {g.node_features.shape[0]}, "
          f"Features: {g.node_features.shape[1]}")
    print(f"  Edges: {g.edge_index.shape[1]}, "
          f"Edge features: {g.edge_features.shape[1]}")
    print(f"  Targets: {g.node_targets.shape[0]}")
    print(f"  Global: {g.global_features}")

    # Test physics loss
    loss = physics_informed_loss(
        d_next + np.random.randn(n_nodes)*0.001, d_next, g, {})
    print(f"  Physics loss: {loss}")

    # Temporal sequence
    d_hist = [d_field + i*0.01*np.random.rand(n_nodes) for i in range(5)]
    u_hist = [u_field + i*0.01*np.random.rand(2*n_nodes) for i in range(5)]
    graphs = build_temporal_graph_sequence(mesh_nodes, mesh_tets, d_hist, u_hist)
    print(f"  Temporal graphs: {len(graphs)}")

    save_graph_dataset(graphs, 'gnn_test_output')
    print("  GNN interface ready!")
