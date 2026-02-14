#!/usr/bin/env python3
"""Perturbation network generation for ligand RBFE calculations.

Simplified port of PyAutoFEP/generate_perturbation_map.py.
Supports both user-defined link files and automatic network generation
using MCS-based scoring.
"""

import itertools
from math import exp
from statistics import median
from typing import Dict, List, Optional, Tuple

try:
    import networkx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

import rdkit.Chem

from atom_mapping import find_mcs


def read_links_file(links_path: str) -> List[Tuple[str, str]]:
    """Read a user-defined perturbation links file.

    File format: one pair per line, "ligand_A ligand_B" (whitespace separated).
    Lines starting with '#' are comments.

    Parameters
    ----------
    links_path : str
        Path to links file.

    Returns
    -------
    list of (str, str)
        List of (ligand_name_A, ligand_name_B) pairs.
    """
    pairs = []
    with open(links_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def compute_perturbation_costs(molecules: Dict[str, rdkit.Chem.Mol],
                               pairs: Optional[List[Tuple[str, str]]] = None,
                               use_hs: bool = False) -> Dict[Tuple[str, str], int]:
    """Compute perturbation cost (number of perturbed atoms) for each pair.

    Parameters
    ----------
    molecules : dict
        {ligand_name: rdkit.Chem.Mol} dictionary.
    pairs : list of (str, str) or None
        Specific pairs to evaluate. If None, all pairs are evaluated.
    use_hs : bool
        Count hydrogen atoms in perturbation cost.

    Returns
    -------
    dict
        {(name_A, name_B): perturbed_atoms} dictionary.
    """
    if pairs is None:
        names = list(molecules.keys())
        pairs = [(a, b) for a, b in itertools.combinations(names, 2)]

    costs = {}
    for name_a, name_b in pairs:
        mol_a = molecules[name_a]
        mol_b = molecules[name_b]

        try:
            mcs = find_mcs([rdkit.Chem.RemoveHs(mol_a), rdkit.Chem.RemoveHs(mol_b)])
        except RuntimeError:
            # MCS failed; set very high cost
            costs[(name_a, name_b)] = 999
            continue

        core_mol = rdkit.Chem.MolFromSmarts(mcs.smartsString)
        if core_mol is None:
            core_mol = rdkit.Chem.MolFromSmiles(mcs.smartsString)

        if core_mol is None:
            costs[(name_a, name_b)] = 999
            continue

        if use_hs:
            n_core = core_mol.GetNumAtoms()
            n_a = mol_a.GetNumAtoms()
            n_b = mol_b.GetNumAtoms()
        else:
            n_core = core_mol.GetNumHeavyAtoms()
            n_a = mol_a.GetNumHeavyAtoms()
            n_b = mol_b.GetNumHeavyAtoms()

        perturbed = (n_a - n_core) + (n_b - n_core)
        costs[(name_a, name_b)] = perturbed

    return costs


def build_thermograph(molecules: Dict[str, rdkit.Chem.Mol],
                      pairs: Optional[List[Tuple[str, str]]] = None) -> 'networkx.Graph':
    """Build a weighted graph of perturbation costs.

    Parameters
    ----------
    molecules : dict
        {name: mol} dictionary.
    pairs : list of (str, str) or None
        Pairs to include. If None, all combinations.

    Returns
    -------
    networkx.Graph
        Graph with edges weighted by perturbation cost.
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for automatic network generation")

    G = networkx.Graph()
    G.add_nodes_from(molecules.keys())

    costs = compute_perturbation_costs(molecules, pairs)

    for (name_a, name_b), perturbed in costs.items():
        if perturbed == 0:
            continue  # Skip identical molecules
        G.add_edge(name_a, name_b, perturbed_atoms=perturbed, weight=perturbed)

    return G


def generate_optimal_network(molecules: Dict[str, rdkit.Chem.Mol],
                             max_edges_per_node: int = 3,
                             max_perturbed_atoms: int = 10) -> List[Tuple[str, str]]:
    """Generate an optimal perturbation network.

    Uses a greedy approach: start with minimum spanning tree and add
    edges to improve connectivity and redundancy.

    Parameters
    ----------
    molecules : dict
        {name: mol} dictionary.
    max_edges_per_node : int
        Maximum number of edges per node.
    max_perturbed_atoms : int
        Maximum number of perturbed atoms per edge.

    Returns
    -------
    list of (str, str)
        Selected perturbation pairs.
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for automatic network generation")

    G = build_thermograph(molecules)

    # Filter edges with too many perturbed atoms
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                       if d['perturbed_atoms'] > max_perturbed_atoms]
    G.remove_edges_from(edges_to_remove)

    if not networkx.is_connected(G):
        print("Warning: graph is not connected after filtering. "
              "Some ligands may not be connected.")

    # Start with minimum spanning tree
    try:
        mst = networkx.minimum_spanning_tree(G, weight='weight')
    except networkx.exception.NetworkXError:
        print("Warning: Could not compute MST. Using all available edges.")
        return list(G.edges())

    selected_edges = set(mst.edges())

    # Add redundant edges (lowest cost) up to max_edges_per_node
    all_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
    for u, v, data in all_edges:
        if (u, v) in selected_edges or (v, u) in selected_edges:
            continue
        deg_u = sum(1 for e in selected_edges if u in e)
        deg_v = sum(1 for e in selected_edges if v in e)
        if deg_u < max_edges_per_node and deg_v < max_edges_per_node:
            selected_edges.add((u, v))

    return list(selected_edges)


def generate_star_network(molecules: Dict[str, rdkit.Chem.Mol],
                          center: str) -> List[Tuple[str, str]]:
    """Generate a star-shaped perturbation network.

    Parameters
    ----------
    molecules : dict
        {name: mol} dictionary.
    center : str
        Name of the center molecule.

    Returns
    -------
    list of (str, str)
        Perturbation pairs.
    """
    if center not in molecules:
        raise ValueError(f"Center molecule '{center}' not found in molecules")

    return [(center, name) for name in molecules if name != center]
