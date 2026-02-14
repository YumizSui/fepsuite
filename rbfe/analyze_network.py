#!/usr/bin/env python3
"""Network-level analysis for ligand RBFE calculations.

Collects per-edge BAR/MBAR results, performs cycle closure analysis,
and computes relative binding free energies referenced to a chosen ligand.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# in kcal/mol/K
GASCONSTANT_KCAL = 0.0019872036
# in kJ/mol/K
GASCONSTANT = 0.008314472


def load_edge_result(edge_dir: str) -> Optional[Tuple[float, float]]:
    """Load BAR result from an edge directory.

    Looks for results-normalsplit.pickle in the bar/ subdirectory.

    Returns
    -------
    (mean_ddg, stderr) in kcal/mol, or None if not available.
    """
    pickle_path = os.path.join(edge_dir, "bar", "results-normalsplit.pickle")
    if not os.path.exists(pickle_path):
        return None

    with open(pickle_path, 'rb') as fh:
        results = pickle.load(fh)

    if not results:
        return None

    # results is list of (t0, t1, barres_in_beta_units)
    # Convert to kcal/mol: barres * kT * (1/4.184)
    temp = 300.0  # default
    kcal_of_kJ = 1.0 / 4.184
    vals = [x * GASCONSTANT * temp * kcal_of_kJ for (_, _, x) in results]
    mean = np.mean(vals)
    stderr = 0.0
    if len(vals) > 2:
        stderr = np.std(vals, ddof=1) / np.sqrt(len(vals) - 1)

    return (mean, stderr)


def load_all_edges(edges_file: str, temp: float = 300.0) -> dict:
    """Load all edge results from edges.txt.

    Parameters
    ----------
    edges_file : str
        Path to edges.txt (tab-separated: ligA, ligB, holo_dir, ref_dir).
    temp : float
        Temperature in K.

    Returns
    -------
    dict with keys:
        'edges': list of (ligA, ligB, ddg_holo, ddg_ref, ddg_bind, stderr)
        'names': set of ligand names
    """
    edges = []
    names = set()

    with open(edges_file) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            lig_a, lig_b, holo_dir, ref_dir = parts[:4]
            names.add(lig_a)
            names.add(lig_b)

            holo_result = load_edge_result(holo_dir)
            ref_result = load_edge_result(ref_dir)

            if holo_result is None or ref_result is None:
                print(f"Warning: Missing results for edge {lig_a} -> {lig_b}")
                continue

            ddg_holo, err_holo = holo_result
            ddg_ref, err_ref = ref_result

            # DDG_bind = DDG_holo - DDG_ref
            ddg_bind = ddg_holo - ddg_ref
            stderr = np.sqrt(err_holo**2 + err_ref**2)

            edges.append({
                'lig_a': lig_a,
                'lig_b': lig_b,
                'ddg_holo': ddg_holo,
                'ddg_ref': ddg_ref,
                'ddg_bind': ddg_bind,
                'stderr': stderr,
            })

    return {'edges': edges, 'names': names}


def compute_relative_dg(edges: list, reference: str) -> Dict[str, Tuple[float, float]]:
    """Compute relative binding free energies from network edges.

    Uses shortest-path approach on the network graph.

    Parameters
    ----------
    edges : list of dicts
        Edge data from load_all_edges.
    reference : str
        Reference ligand name (DDG = 0).

    Returns
    -------
    dict of {ligand_name: (relative_ddg, uncertainty)}
    """
    if not HAS_NETWORKX:
        # Simple accumulation without networkx
        return _compute_relative_dg_simple(edges, reference)

    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge['lig_a'], edge['lig_b'],
                    ddg=edge['ddg_bind'], stderr=edge['stderr'])
        G.add_edge(edge['lig_b'], edge['lig_a'],
                    ddg=-edge['ddg_bind'], stderr=edge['stderr'])

    if reference not in G:
        raise ValueError(f"Reference ligand '{reference}' not in network")

    results = {reference: (0.0, 0.0)}

    for node in G.nodes():
        if node == reference:
            continue
        try:
            path = nx.shortest_path(G, reference, node, weight=None)
        except nx.NetworkXNoPath:
            print(f"Warning: No path from {reference} to {node}")
            continue

        ddg_total = 0.0
        var_total = 0.0
        for i in range(len(path) - 1):
            edge_data = G[path[i]][path[i+1]]
            ddg_total += edge_data['ddg']
            var_total += edge_data['stderr']**2

        results[node] = (ddg_total, np.sqrt(var_total))

    return results


def _compute_relative_dg_simple(edges, reference):
    """Simple relative DDG computation without networkx."""
    # Build adjacency
    adj = {}
    for edge in edges:
        a, b = edge['lig_a'], edge['lig_b']
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append((b, edge['ddg_bind'], edge['stderr']))
        adj[b].append((a, -edge['ddg_bind'], edge['stderr']))

    # BFS from reference
    results = {reference: (0.0, 0.0)}
    queue = [reference]
    visited = {reference}

    while queue:
        current = queue.pop(0)
        if current not in adj:
            continue
        for neighbor, ddg, stderr in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                cur_ddg, cur_var = results[current]
                results[neighbor] = (cur_ddg + ddg, np.sqrt(cur_var**2 + stderr**2))
                queue.append(neighbor)

    return results


def cycle_closure_analysis(edges: list) -> List[dict]:
    """Check thermodynamic cycle closure.

    For each cycle in the network, compute the sum of DDG values
    around the cycle (should be ~0 for consistent results).

    Returns list of cycle info dicts.
    """
    if not HAS_NETWORKX:
        print("Warning: NetworkX not available, skipping cycle closure analysis")
        return []

    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge['lig_a'], edge['lig_b'],
                    ddg=edge['ddg_bind'], stderr=edge['stderr'])

    cycles = nx.cycle_basis(G)
    cycle_results = []

    for cycle in cycles:
        ddg_sum = 0.0
        var_sum = 0.0
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            edge_data = G[a][b]
            # Need direction: check which way the original edge goes
            forward = any(e['lig_a'] == a and e['lig_b'] == b for e in edges)
            if forward:
                ddg_sum += edge_data['ddg']
            else:
                ddg_sum -= edge_data['ddg']
            var_sum += edge_data['stderr']**2

        cycle_results.append({
            'cycle': cycle,
            'closure_error': ddg_sum,
            'uncertainty': np.sqrt(var_sum),
        })

    return cycle_results


def main():
    parser = argparse.ArgumentParser(description="Analyze RBFE network results")
    parser.add_argument("--edges", default="edges.txt",
                        help="Path to edges.txt file")
    parser.add_argument("--reference", default=None,
                        help="Reference ligand name (default: first ligand)")
    parser.add_argument("--temp", default=300.0, type=float,
                        help="Temperature (K)")
    parser.add_argument("--experimental", default=None,
                        help="File with experimental values (name dG_exp)")

    args = parser.parse_args()

    data = load_all_edges(args.edges, args.temp)

    if not data['edges']:
        print("No results found. Make sure BAR analysis has been run.")
        sys.exit(1)

    print(f"Loaded {len(data['edges'])} edge results")
    print()

    # Print per-edge results
    print("Per-edge results (kcal/mol):")
    print(f"{'Edge':<30s} {'DDG_holo':>10s} {'DDG_ref':>10s} {'DDG_bind':>10s} {'stderr':>8s}")
    print("-" * 68)
    for edge in data['edges']:
        label = f"{edge['lig_a']} -> {edge['lig_b']}"
        print(f"{label:<30s} {edge['ddg_holo']:>10.2f} {edge['ddg_ref']:>10.2f} "
              f"{edge['ddg_bind']:>10.2f} {edge['stderr']:>8.2f}")

    # Relative DDG
    reference = args.reference
    if reference is None:
        reference = data['edges'][0]['lig_a']

    print(f"\nRelative binding free energies (reference: {reference}):")
    rel_dg = compute_relative_dg(data['edges'], reference)
    print(f"{'Ligand':<20s} {'DDG':>10s} {'stderr':>8s}")
    print("-" * 38)
    for name in sorted(rel_dg.keys()):
        ddg, stderr = rel_dg[name]
        print(f"{name:<20s} {ddg:>10.2f} {stderr:>8.2f}")

    # Cycle closure
    cycle_results = cycle_closure_analysis(data['edges'])
    if cycle_results:
        print(f"\nCycle closure analysis:")
        for cr in cycle_results:
            cycle_str = " -> ".join(cr['cycle'] + [cr['cycle'][0]])
            print(f"  {cycle_str}: closure = {cr['closure_error']:.2f} "
                  f"(+/- {cr['uncertainty']:.2f}) kcal/mol")

    # Compare with experimental if available
    if args.experimental:
        print(f"\nComparison with experimental values:")
        exp_values = {}
        with open(args.experimental) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    exp_values[parts[0]] = float(parts[1])

        calc_vals = []
        exp_vals = []
        print(f"{'Ligand':<20s} {'DDG_calc':>10s} {'DDG_exp':>10s} {'Diff':>8s}")
        print("-" * 48)
        for name in sorted(rel_dg.keys()):
            if name in exp_values:
                ddg_calc, _ = rel_dg[name]
                ddg_exp = exp_values[name]
                diff = ddg_calc - ddg_exp
                print(f"{name:<20s} {ddg_calc:>10.2f} {ddg_exp:>10.2f} {diff:>8.2f}")
                calc_vals.append(ddg_calc)
                exp_vals.append(ddg_exp)

        if len(calc_vals) > 1:
            calc_arr = np.array(calc_vals)
            exp_arr = np.array(exp_vals)
            rmse = np.sqrt(np.mean((calc_arr - exp_arr)**2))
            mae = np.mean(np.abs(calc_arr - exp_arr))
            correlation = np.corrcoef(calc_arr, exp_arr)[0, 1]
            print(f"\nRMSE: {rmse:.2f} kcal/mol")
            print(f"MAE:  {mae:.2f} kcal/mol")
            print(f"R:    {correlation:.3f}")


if __name__ == "__main__":
    main()
