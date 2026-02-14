#!/usr/bin/env python3
"""Atom mapping between two ligands using MCS (Maximum Common Substructure).

Ported and simplified from PyAutoFEP/merge_topologies.py.
Uses RDKit MCS to find common atoms between two ligands.
"""

import re
from collections import namedtuple
from typing import List, Optional, Tuple

import rdkit.Chem
import rdkit.Chem.rdFMCS
import rdkit.Chem.AllChem
from rdkit.Chem.AllChem import AlignMol, ConstrainedEmbed

MCSResult = namedtuple('MCSResult', ['smartsString', 'num_atoms', 'num_bonds', 'canceled'])


def find_mcs(mol_list: list, match_valences: bool = True,
             ring_matches_ring_only: bool = True,
             complete_rings_only: bool = True,
             timeout: int = 3600) -> MCSResult:
    """Find Maximum Common Substructure between molecules.

    Uses a two-pass approach: first on heavy atoms with hashed isotopes
    for speed, then refines with full hydrogen matching if needed.

    Parameters
    ----------
    mol_list : list of rdkit.Chem.Mol
        Molecules to find MCS between.
    match_valences : bool
        Require matching valences.
    ring_matches_ring_only : bool
        Ring atoms only match ring atoms.
    complete_rings_only : bool
        Only include complete rings in MCS.
    timeout : int
        Timeout in seconds for MCS search.

    Returns
    -------
    MCSResult
        Named tuple with smartsString, num_atoms, num_bonds, canceled.
    """
    # Check if molecules are identical
    smiles_set = set(rdkit.Chem.MolToSmiles(m) for m in mol_list)
    if len(smiles_set) == 1:
        mol = mol_list[0]
        return MCSResult(
            rdkit.Chem.MolToSmiles(mol),
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            canceled=False,
        )

    # First pass: MCS on heavy atoms with isotope hashing
    altered_mols = [rdkit.Chem.RemoveHs(rdkit.Chem.Mol(m)) for m in mol_list]

    # Hash atoms using hybridization, atomic number and ring membership
    for mol in altered_mols:
        for atom in mol.GetAtoms():
            isotope = (1000 * int(atom.IsInRing())
                       + 100 * int(atom.GetHybridization())
                       + atom.GetAtomicNum())
            atom.SetIsotope(isotope)

    mcs_result = rdkit.Chem.rdFMCS.FindMCS(
        altered_mols,
        completeRingsOnly=complete_rings_only,
        ringMatchesRingOnly=ring_matches_ring_only,
        matchChiralTag=False,
        atomCompare=rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes,
        timeout=timeout,
    )

    if mcs_result.canceled or mcs_result.numAtoms == 0:
        raise RuntimeError(f"Failed to calculate MCS. Try increasing timeout (current: {timeout}s)")

    # Convert hashed SMARTS back to real atoms
    tempmol = rdkit.Chem.MolFromSmarts(mcs_result.smartsString)
    if tempmol is None:
        raise RuntimeError(f"FindMCS returned invalid SMARTS: {mcs_result.smartsString}")
    for atom in tempmol.GetAtoms():
        atom.SetAtomicNum(atom.GetIsotope() % 100)
        atom.SetIsotope(0)

    # Get SMILES of common core from first molecule
    core_mol = rdkit.Chem.MolFromSmarts(mcs_result.smartsString)
    match = altered_mols[0].GetSubstructMatch(core_mol)
    if not match:
        raise RuntimeError("Failed to match MCS to first molecule")

    template_mol = rdkit.Chem.RemoveHs(rdkit.Chem.Mol(mol_list[0]))
    common_smiles = rdkit.Chem.MolFragmentToSmiles(
        template_mol, atomsToUse=match, isomericSmiles=True, canonical=False
    )

    # Construct mol and sanitize
    common_mol = rdkit.Chem.MolFromSmiles(common_smiles, sanitize=False)
    if common_mol is not None:
        for atom in common_mol.GetAtoms():
            if atom.GetFormalCharge() < 0:
                atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
        common_mol.UpdatePropertyCache()
        try:
            rdkit.Chem.SanitizeMol(common_mol)
        except Exception:
            pass  # Proceed even if sanitization partially fails

    # For molecules without explicit Hs, return first MCS
    if all(m.GetNumAtoms() == m.GetNumHeavyAtoms() for m in mol_list):
        smarts = rdkit.Chem.MolToSmarts(common_mol) if common_mol else common_smiles
        return MCSResult(smarts, mcs_result.numAtoms, mcs_result.numBonds, False)

    # Second pass: use first MCS info to rehash for refined search
    altered_mols2 = [rdkit.Chem.Mol(m) for m in mol_list]

    if common_mol is not None:
        _adjust_query_properties(common_mol)
        for mol in altered_mols2:
            matches_list = mol.GetSubstructMatches(common_mol, uniquify=False, maxMatches=1000)
            if matches_list:
                # Mark atoms in core
                for each_match in matches_list:
                    for core_idx, mol_idx in enumerate(each_match):
                        mol.GetAtomWithIdx(mol_idx).SetIsotope(core_idx + 100)

                # Mark atoms not in core with unique isotopes
                unmatch_iso = 1000
                for atom in mol.GetAtoms():
                    if atom.GetIsotope() == 0:
                        if atom.GetAtomicNum() == 1:
                            atom.SetIsotope(1)
                        else:
                            atom.SetIsotope(unmatch_iso)
                            unmatch_iso += 1

    mcs_result2 = rdkit.Chem.rdFMCS.FindMCS(
        altered_mols2,
        completeRingsOnly=complete_rings_only,
        ringMatchesRingOnly=ring_matches_ring_only,
        matchChiralTag=False,
        atomCompare=rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes,
        timeout=timeout,
    )

    if mcs_result2.canceled or mcs_result2.numAtoms == 0:
        # Fall back to first MCS result
        smarts = rdkit.Chem.MolToSmarts(common_mol) if common_mol else common_smiles
        return MCSResult(smarts, mcs_result.numAtoms, mcs_result.numBonds, False)

    core_mol2 = rdkit.Chem.MolFromSmarts(mcs_result2.smartsString)
    _adjust_query_properties(core_mol2)
    match2 = altered_mols2[0].GetSubstructMatch(core_mol2)
    if match2:
        final_smiles = rdkit.Chem.MolFragmentToSmiles(
            mol_list[0], atomsToUse=match2, isomericSmiles=True, canonical=True
        )
    else:
        final_smiles = common_smiles

    return MCSResult(final_smiles, mcs_result2.numAtoms, mcs_result2.numBonds, False)


def get_atom_map(mol_a: rdkit.Chem.Mol, mol_b: rdkit.Chem.Mol,
                 mcs_smarts: Optional[str] = None) -> List[Tuple[int, int]]:
    """Get atom mapping between two molecules based on MCS.

    Parameters
    ----------
    mol_a : rdkit.Chem.Mol
        First molecule.
    mol_b : rdkit.Chem.Mol
        Second molecule.
    mcs_smarts : str or None
        Pre-computed MCS SMARTS. If None, computed automatically.

    Returns
    -------
    list of (int, int)
        List of (atom_idx_in_A, atom_idx_in_B) pairs.
    """
    if mcs_smarts is None:
        mcs = find_mcs([mol_a, mol_b])
        mcs_smarts = mcs.smartsString

    core_mol = rdkit.Chem.MolFromSmarts(mcs_smarts)
    if core_mol is None:
        core_mol = rdkit.Chem.MolFromSmiles(mcs_smarts)
    if core_mol is None:
        raise ValueError(f"Cannot parse MCS: {mcs_smarts}")

    _adjust_query_properties(core_mol)

    match_a = mol_a.GetSubstructMatch(core_mol)
    match_b = mol_b.GetSubstructMatch(core_mol)

    if not match_a or not match_b:
        raise ValueError("Could not match MCS to one or both molecules")

    atom_map = list(zip(match_a, match_b))
    return atom_map


def align_ligand_to_reference(mobile_mol: rdkit.Chem.Mol,
                              ref_mol: rdkit.Chem.Mol,
                              atom_map: List[Tuple[int, int]]) -> rdkit.Chem.Mol:
    """Align mobile molecule to reference using atom map.

    Parameters
    ----------
    mobile_mol : rdkit.Chem.Mol
        Molecule to be aligned (will be modified in place).
    ref_mol : rdkit.Chem.Mol
        Reference molecule.
    atom_map : list of (int, int)
        Atom map as (mobile_idx, ref_idx) pairs.

    Returns
    -------
    rdkit.Chem.Mol
        Aligned molecule.
    """
    # AlignMol expects (probe, ref) atom map as list of (probe_idx, ref_idx)
    AlignMol(mobile_mol, ref_mol, atomMap=atom_map)
    return mobile_mol


def constrained_embed_ligand(mol: rdkit.Chem.Mol, core: rdkit.Chem.Mol,
                             atom_map: Optional[list] = None) -> rdkit.Chem.Mol:
    """Embed molecule constrained to core using ConstrainedEmbed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to embed.
    core : rdkit.Chem.Mol
        Core (reference) molecule with 3D coordinates.
    atom_map : list or None
        If None, automatically detected.

    Returns
    -------
    rdkit.Chem.Mol
        Embedded molecule.
    """
    mol_copy = rdkit.Chem.Mol(mol)
    try:
        ConstrainedEmbed(mol_copy, core, enforceChirality=True)
    except ValueError:
        # Fallback: force embed with relaxed constraints
        rdkit.Chem.AllChem.EmbedMolecule(
            mol_copy, randomSeed=42, useRandomCoords=True,
            ignoreSmoothingFailures=True
        )
        if atom_map:
            AlignMol(mol_copy, core, atomMap=[(a, b) for a, b in atom_map])
    return mol_copy


def _adjust_query_properties(mol):
    """Adjust query properties for substructure matching compatibility."""
    if mol is None:
        return mol
    params = rdkit.Chem.AdjustQueryParameters.NoAdjustments()
    params.makeAtomsGeneric = False
    params.makeBondsGeneric = True
    try:
        mol = rdkit.Chem.AdjustQueryProperties(mol, params)
    except Exception:
        pass
    return mol
