#!/usr/bin/env python3
"""Dual-topology generation for ligand RBFE.

Simplified port of PyAutoFEP/merge_topologies.py:merge_topologies().
Creates a merged (dual-state) ligand topology where atoms in the common
core share coordinates, and atoms unique to each state become dummy atoms
in the other state.
"""

import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import rdkit.Chem
from rdkit.Chem.AllChem import AlignMol

from atom_mapping import find_mcs, get_atom_map


def merge_ligand_topologies(mol_a: rdkit.Chem.Mol, mol_b: rdkit.Chem.Mol,
                            itp_a: str, itp_b: str,
                            output_itp: str, output_gro: str,
                            mcs_smarts: Optional[str] = None) -> dict:
    """Create a dual-topology ITP for ligand A->B transformation.

    Parameters
    ----------
    mol_a : rdkit.Chem.Mol
        Molecule A (with 3D coordinates).
    mol_b : rdkit.Chem.Mol
        Molecule B (with 3D coordinates).
    itp_a : str
        Path to ITP topology of molecule A.
    itp_b : str
        Path to ITP topology of molecule B.
    output_itp : str
        Path to write merged dual-state ITP.
    output_gro : str
        Path to write merged GRO coordinates.
    mcs_smarts : str or None
        Pre-computed MCS SMARTS. If None, computed.

    Returns
    -------
    dict
        {'itp': output_itp, 'gro': output_gro, 'atom_map': atom_map,
         'only_in_a': [...], 'only_in_b': [...], 'merged_name': str}
    """
    # Get atom mapping
    atom_map = get_atom_map(mol_a, mol_b, mcs_smarts)
    a_in_core = set(a for a, b in atom_map)
    b_in_core = set(b for a, b in atom_map)

    only_in_a = [i for i in range(mol_a.GetNumAtoms()) if i not in a_in_core]
    only_in_b = [i for i in range(mol_b.GetNumAtoms()) if i not in b_in_core]

    # Parse topology files
    top_a = _parse_itp(itp_a)
    top_b = _parse_itp(itp_b)

    mol_name_a = top_a['moleculetype_name']
    mol_name_b = top_b['moleculetype_name']
    merged_name = f"{mol_name_a}~{mol_name_b}"

    # Build index mapping: B atom idx -> merged atom idx
    # Core atoms from B map to corresponding A atom indices
    # B-only atoms are appended after all A atoms
    n_atoms_a = len(top_a['atoms'])
    b_to_merged = {}
    for a_idx, b_idx in atom_map:
        b_to_merged[b_idx] = a_idx  # 0-indexed
    next_idx = n_atoms_a
    for b_idx in only_in_b:
        b_to_merged[b_idx] = next_idx
        next_idx += 1

    total_atoms = next_idx

    # Build merged atom lines
    merged_atoms = []
    for i, atom_line in enumerate(top_a['atoms']):
        # Core atoms: perturb charges
        if i in a_in_core:
            b_idx = [b for a, b in atom_map if a == i][0]
            b_atom = top_b['atoms'][b_idx]
            merged_atom = _merge_atom_line(atom_line, b_atom, i + 1, state='core')
        else:
            # A-only atoms: become dummy in state B
            merged_atom = _merge_atom_line(atom_line, None, i + 1, state='A_only')
        merged_atoms.append(merged_atom)

    # B-only atoms: dummy in state A, real in state B
    for b_idx in only_in_b:
        b_atom = top_b['atoms'][b_idx]
        new_idx = b_to_merged[b_idx]
        merged_atom = _merge_atom_line(None, b_atom, new_idx + 1, state='B_only')
        merged_atoms.append(merged_atom)

    # Build merged bonded interactions
    merged_bonds = list(top_a['bonds'])
    merged_pairs = list(top_a['pairs'])
    merged_angles = list(top_a['angles'])
    merged_dihedrals = list(top_a['dihedrals'])

    # Add B-only bonded terms (involving at least one B-only atom)
    for section_name, a_terms, b_terms in [
        ('bonds', merged_bonds, top_b['bonds']),
        ('pairs', merged_pairs, top_b['pairs']),
        ('angles', merged_angles, top_b['angles']),
        ('dihedrals', merged_dihedrals, top_b['dihedrals']),
    ]:
        for term in b_terms:
            atoms_in_term = term['atoms']
            # Include if any atom is B-only
            if any(a in only_in_b for a in atoms_in_term):
                new_term = deepcopy(term)
                new_term['atoms'] = [b_to_merged[a] for a in atoms_in_term]
                new_term['comment'] = f'; Added from topology B'
                a_terms.append(new_term)

    # Build exclusions between A-only and B-only atoms
    exclusions = []
    for a_idx in only_in_a:
        for b_idx in only_in_b:
            exclusions.append((a_idx, b_to_merged[b_idx]))

    # Write merged ITP
    _write_merged_itp(output_itp, merged_name, merged_atoms,
                      merged_bonds, merged_pairs, merged_angles,
                      merged_dihedrals, exclusions, top_a, top_b)

    # Align molecule B to A and write merged GRO
    mol_b_aligned = rdkit.Chem.Mol(mol_b)
    align_map = [(b, a) for a, b in atom_map]
    try:
        AlignMol(mol_b_aligned, mol_a, atomMap=align_map)
    except Exception:
        print("Warning: alignment failed, using original coordinates for B")
        mol_b_aligned = mol_b

    _write_merged_gro(output_gro, merged_name, mol_a, mol_b_aligned,
                      atom_map, only_in_b, top_a, top_b)

    return {
        'itp': output_itp,
        'gro': output_gro,
        'atom_map': atom_map,
        'only_in_a': only_in_a,
        'only_in_b': only_in_b,
        'merged_name': merged_name,
    }


def _parse_itp(itp_path: str) -> dict:
    """Parse a GROMACS ITP file into structured data."""
    data = {
        'moleculetype_name': '',
        'atoms': [],      # list of dicts
        'bonds': [],
        'pairs': [],
        'angles': [],
        'dihedrals': [],
        'atomtypes': [],   # raw lines
        'raw_lines': [],
    }

    section = None
    with open(itp_path) as fh:
        for line in fh:
            raw = line.rstrip('\n')
            data['raw_lines'].append(raw)

            stripped = line.split(';')[0].strip()
            comment = ''
            if ';' in line:
                comment = ';' + line.split(';', 1)[1].rstrip('\n')

            if not stripped:
                continue
            if stripped.startswith('['):
                section = stripped.strip('[] \t').lower()
                continue
            if stripped.startswith('#'):
                continue

            parts = stripped.split()

            if section == 'moleculetype':
                data['moleculetype_name'] = parts[0]

            elif section == 'atomtypes':
                data['atomtypes'].append(line)

            elif section == 'atoms':
                # nr  type  resnr  resname  atom  cgnr  charge  mass
                atom = {
                    'nr': int(parts[0]),
                    'type': parts[1],
                    'resnr': int(parts[2]),
                    'resname': parts[3],
                    'atom': parts[4],
                    'cgnr': int(parts[5]),
                    'charge': float(parts[6]),
                    'mass': float(parts[7]) if len(parts) > 7 else None,
                    'typeB': parts[8] if len(parts) > 8 else None,
                    'chargeB': float(parts[9]) if len(parts) > 9 else None,
                    'massB': float(parts[10]) if len(parts) > 10 else None,
                    'comment': comment,
                }
                data['atoms'].append(atom)

            elif section == 'bonds':
                atom_indices = [int(parts[i]) - 1 for i in range(2)]
                data['bonds'].append({
                    'atoms': atom_indices,
                    'func': int(parts[2]) if len(parts) > 2 else 1,
                    'params': parts[3:] if len(parts) > 3 else [],
                    'comment': comment,
                })

            elif section == 'pairs':
                atom_indices = [int(parts[i]) - 1 for i in range(2)]
                data['pairs'].append({
                    'atoms': atom_indices,
                    'func': int(parts[2]) if len(parts) > 2 else 1,
                    'params': parts[3:] if len(parts) > 3 else [],
                    'comment': comment,
                })

            elif section == 'angles':
                atom_indices = [int(parts[i]) - 1 for i in range(3)]
                data['angles'].append({
                    'atoms': atom_indices,
                    'func': int(parts[3]) if len(parts) > 3 else 1,
                    'params': parts[4:] if len(parts) > 4 else [],
                    'comment': comment,
                })

            elif section == 'dihedrals':
                atom_indices = [int(parts[i]) - 1 for i in range(4)]
                data['dihedrals'].append({
                    'atoms': atom_indices,
                    'func': int(parts[4]) if len(parts) > 4 else 1,
                    'params': parts[5:] if len(parts) > 5 else [],
                    'comment': comment,
                })

    return data


def _merge_atom_line(atom_a, atom_b, nr, state):
    """Create a merged atom entry.

    state: 'core' - shared atom, perturb charges
           'A_only' - dummy in state B
           'B_only' - dummy in state A
    """
    if state == 'core':
        return {
            'nr': nr,
            'type': atom_a['type'],
            'resnr': atom_a['resnr'],
            'resname': atom_a['resname'],
            'atom': atom_a['atom'],
            'cgnr': nr,
            'charge': atom_a['charge'],
            'mass': atom_a['mass'],
            'typeB': atom_b['type'],
            'chargeB': atom_b['charge'],
            'massB': atom_b['mass'],
        }
    elif state == 'A_only':
        return {
            'nr': nr,
            'type': atom_a['type'],
            'resnr': atom_a['resnr'],
            'resname': atom_a['resname'],
            'atom': atom_a['atom'],
            'cgnr': nr,
            'charge': atom_a['charge'],
            'mass': atom_a['mass'],
            'typeB': 'DU',
            'chargeB': 0.0,
            'massB': atom_a['mass'],
        }
    elif state == 'B_only':
        return {
            'nr': nr,
            'type': 'DU',
            'resnr': atom_b['resnr'],
            'resname': atom_b['resname'],
            'atom': atom_b['atom'],
            'cgnr': nr,
            'charge': 0.0,
            'mass': atom_b['mass'],
            'typeB': atom_b['type'],
            'chargeB': atom_b['charge'],
            'massB': atom_b['mass'],
        }


def _write_merged_itp(output_path, name, atoms, bonds, pairs, angles,
                      dihedrals, exclusions, top_a, top_b):
    """Write the merged dual-topology ITP file."""

    # Collect atomtypes from both topologies
    atomtype_set = set()
    all_atomtype_lines = []
    for at_line in top_a.get('atomtypes', []) + top_b.get('atomtypes', []):
        key = at_line.split()[0] if at_line.strip() else ''
        if key and key not in atomtype_set:
            atomtype_set.add(key)
            all_atomtype_lines.append(at_line)

    with open(output_path, 'w') as ofh:
        # Atomtypes
        if all_atomtype_lines:
            ofh.write('[ atomtypes ]\n')
            # Add dummy atom type
            ofh.write('; Dummy atom type for FEP\n')
            ofh.write(' DU    DU    0  0.000  0.000  A  0.0  0.0\n')
            for line in all_atomtype_lines:
                ofh.write(line)
            ofh.write('\n')
        else:
            ofh.write('[ atomtypes ]\n')
            ofh.write(' DU    DU    0  0.000  0.000  A  0.0  0.0\n')
            ofh.write('\n')

        # Moleculetype
        ofh.write('[ moleculetype ]\n')
        ofh.write(f'; Name    nrexcl\n')
        ofh.write(f' {name}    3\n\n')

        # Atoms
        ofh.write('[ atoms ]\n')
        ofh.write(';  nr  type  resnr  resname  atom  cgnr  charge  mass  typeB  chargeB  massB\n')
        for atom in atoms:
            line = (f"  {atom['nr']:>5d} {atom['type']:>6s} {atom['resnr']:>5d} "
                    f"{atom['resname']:>5s} {atom['atom']:>5s} {atom['cgnr']:>5d} "
                    f"{atom['charge']:>10.4f}")
            if atom.get('mass') is not None:
                line += f" {atom['mass']:>10.4f}"
            if atom.get('typeB') is not None:
                mass_b = atom.get('massB', atom.get('mass', 0.0))
                if mass_b is None:
                    mass_b = 0.0
                line += (f"  {atom['typeB']:>6s} {atom['chargeB']:>10.4f}"
                         f" {mass_b:>10.4f}")
            ofh.write(line + '\n')
        ofh.write('\n')

        # Bonds
        ofh.write('[ bonds ]\n')
        for term in bonds:
            indices = ' '.join(str(a + 1) for a in term['atoms'])
            params = ' '.join(str(p) for p in term['params'])
            comment = term.get('comment', '')
            ofh.write(f"  {indices}  {term['func']}  {params}  {comment}\n")
        ofh.write('\n')

        # Pairs
        if pairs:
            ofh.write('[ pairs ]\n')
            for term in pairs:
                indices = ' '.join(str(a + 1) for a in term['atoms'])
                params = ' '.join(str(p) for p in term['params'])
                comment = term.get('comment', '')
                ofh.write(f"  {indices}  {term['func']}  {params}  {comment}\n")
            ofh.write('\n')

        # Angles
        ofh.write('[ angles ]\n')
        for term in angles:
            indices = ' '.join(str(a + 1) for a in term['atoms'])
            params = ' '.join(str(p) for p in term['params'])
            comment = term.get('comment', '')
            ofh.write(f"  {indices}  {term['func']}  {params}  {comment}\n")
        ofh.write('\n')

        # Dihedrals
        if dihedrals:
            ofh.write('[ dihedrals ]\n')
            for term in dihedrals:
                indices = ' '.join(str(a + 1) for a in term['atoms'])
                params = ' '.join(str(p) for p in term['params'])
                comment = term.get('comment', '')
                ofh.write(f"  {indices}  {term['func']}  {params}  {comment}\n")
            ofh.write('\n')

        # Exclusions
        if exclusions:
            ofh.write('[ exclusions ]\n')
            ofh.write('; A-only atoms excluded from B-only atoms\n')
            for a_idx, b_idx in exclusions:
                ofh.write(f"  {a_idx + 1}  {b_idx + 1}\n")
            ofh.write('\n')


def _write_merged_gro(output_path, name, mol_a, mol_b_aligned,
                      atom_map, only_in_b, top_a, top_b):
    """Write merged GRO file with coordinates from A and B-only from B."""
    conf_a = mol_a.GetConformer()
    conf_b = mol_b_aligned.GetConformer()

    atoms_a = top_a['atoms']
    atoms_b = top_b['atoms']

    lines = []
    # All atoms from A
    for i, atom in enumerate(atoms_a):
        pos = conf_a.GetAtomPosition(i)
        lines.append(_format_gro_line(
            atom['resnr'], atom['resname'], atom['atom'],
            len(lines) + 1, pos.x / 10.0, pos.y / 10.0, pos.z / 10.0
        ))

    # B-only atoms
    for b_idx in only_in_b:
        atom = atoms_b[b_idx]
        pos = conf_b.GetAtomPosition(b_idx)
        lines.append(_format_gro_line(
            atom['resnr'], atom['resname'], atom['atom'],
            len(lines) + 1, pos.x / 10.0, pos.y / 10.0, pos.z / 10.0
        ))

    with open(output_path, 'w') as ofh:
        ofh.write(f'Merged {name}\n')
        ofh.write(f'{len(lines)}\n')
        for line in lines:
            ofh.write(line + '\n')
        # Box vectors (dummy, will be replaced during solvation)
        ofh.write('  10.000  10.000  10.000\n')


def _format_gro_line(resnum, resname, atomname, atomnum, x, y, z):
    """Format a GRO file line."""
    return f"{resnum:>5d}{resname:<5s}{atomname:>5s}{atomnum:>5d}{x:8.3f}{y:8.3f}{z:8.3f}"
