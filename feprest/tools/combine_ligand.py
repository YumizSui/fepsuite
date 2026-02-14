#!/usr/bin/env python3
"""Utilities for combining protein and ligand structures/topologies.

Adds ligand coordinates to protein PDB/GRO and ligand ITP to protein topology.
Used by Feature 1 (protein mutation FEP with ligand present).
"""

import os
import re
from typing import Optional


def combine_structure(protein_pdb: str, ligand_gro: str, output_pdb: str):
    """Combine protein PDB/GRO with ligand GRO coordinates.

    Ligand atoms are inserted after protein ATOM/HETATM records and before
    any solvent (SOL/HOH/WAT) or ion records.

    Parameters
    ----------
    protein_pdb : str
        Path to protein structure (PDB or GRO format).
    ligand_gro : str
        Path to ligand coordinates (GRO format from ACPYPE).
    output_pdb : str
        Path to write combined structure (PDB format).
    """
    if protein_pdb.endswith('.gro'):
        protein_atoms = _read_gro_atoms(protein_pdb)
    else:
        protein_atoms = _read_pdb_atoms(protein_pdb)

    ligand_atoms = _read_gro_as_pdb_atoms(ligand_gro)

    # Split protein atoms into pre-solvent and solvent parts
    solvent_names = {'SOL', 'HOH', 'WAT', 'TIP3', 'NA', 'CL', 'K', 'Na', 'Cl', 'SOD', 'CLA'}
    pre_solvent = []
    solvent = []
    in_solvent = False
    for line in protein_atoms:
        resname = line[17:20].strip() if len(line) > 20 else ''
        if resname in solvent_names:
            in_solvent = True
        if in_solvent:
            solvent.append(line)
        else:
            pre_solvent.append(line)

    # Renumber atoms in combined output
    with open(output_pdb, 'w') as ofh:
        atom_serial = 0
        for line in pre_solvent:
            atom_serial += 1
            ofh.write(_renumber_pdb_atom(line, atom_serial) + '\n')
        for line in ligand_atoms:
            atom_serial += 1
            ofh.write(_renumber_pdb_atom(line, atom_serial) + '\n')
        for line in solvent:
            atom_serial += 1
            ofh.write(_renumber_pdb_atom(line, atom_serial) + '\n')
        ofh.write('END\n')


def combine_topology(protein_top: str, ligand_itp: str, ligand_name: str,
                     output_top: str, ligand_top: Optional[str] = None):
    """Add ligand ITP include and molecule entry to protein topology.

    Inserts `#include "ligand.itp"` before `[ system ]` and adds the
    ligand molecule entry in `[ molecules ]`.

    Parameters
    ----------
    protein_top : str
        Path to protein topology file.
    ligand_itp : str
        Path to ligand ITP file to include.
    ligand_name : str
        Molecule name for the ligand (as in ITP's [ moleculetype ]).
    output_top : str
        Path to write combined topology.
    ligand_top : str or None
        If given, extract atomtypes from this TOP file and include them.
    """
    # If ligand_top is provided, extract atomtypes section
    atomtypes_lines = []
    if ligand_top and os.path.isfile(ligand_top):
        atomtypes_lines = _extract_atomtypes(ligand_top)

    # Copy ligand ITP to same directory as output topology for relative includes
    output_dir = os.path.dirname(os.path.abspath(output_top))
    itp_basename = os.path.basename(ligand_itp)
    dest_itp = os.path.join(output_dir, itp_basename)
    if os.path.abspath(ligand_itp) != os.path.abspath(dest_itp):
        import shutil
        shutil.copy2(ligand_itp, dest_itp)

    with open(protein_top) as fh:
        lines = fh.readlines()

    with open(output_top, 'w') as ofh:
        atomtypes_inserted = False
        itp_inserted = False
        in_molecules = False
        molecules_written = False

        for line in lines:
            stripped = line.split(';')[0].strip()
            tokens = stripped.split()

            # Insert atomtypes right after [ defaults ] section data
            if atomtypes_lines and not atomtypes_inserted:
                if tokens == ['[', 'atomtypes', ']']:
                    # There's already an atomtypes section; append our entries
                    ofh.write(line)
                    # Skip to end of existing atomtypes, then append
                    atomtypes_inserted = True
                    continue
                elif tokens == ['[', 'moleculetype', ']'] and not atomtypes_inserted:
                    # No atomtypes section yet; create one before moleculetype
                    ofh.write('[ atomtypes ]\n')
                    for at_line in atomtypes_lines:
                        ofh.write(at_line)
                    ofh.write('\n')
                    atomtypes_inserted = True

            # Insert #include for ligand ITP before [ system ]
            if tokens == ['[', 'system', ']'] and not itp_inserted:
                ofh.write(f'; Include ligand topology\n')
                ofh.write(f'#include "{itp_basename}"\n\n')
                itp_inserted = True

            ofh.write(line)

            # Track [ molecules ] section to append ligand
            if tokens == ['[', 'molecules', ']']:
                in_molecules = True
                molecules_written = False

        # Append ligand to [ molecules ] at the end if not yet done
        # Insert before any solvent entries if possible
        if not molecules_written:
            ofh.write(f'{ligand_name:<20s} 1\n')

    # If atomtypes were not inserted (no defaults or moleculetype found), warn
    if atomtypes_lines and not atomtypes_inserted:
        print("Warning: Could not insert atomtypes into topology")


def combine_topology_inline(protein_top: str, ligand_itp: str,
                            ligand_name: str, output_top: str,
                            ligand_top: Optional[str] = None):
    """Like combine_topology but inserts ligand in correct molecule order.

    The ligand is inserted in [ molecules ] after protein but before solvent.
    This is important for GROMACS to match coordinate order.

    Parameters
    ----------
    protein_top, ligand_itp, ligand_name, output_top, ligand_top
        Same as combine_topology.
    """
    atomtypes_lines = []
    if ligand_top and os.path.isfile(ligand_top):
        atomtypes_lines = _extract_atomtypes(ligand_top)
    elif os.path.isfile(ligand_itp):
        # If no separate TOP file, extract from ITP
        atomtypes_lines = _extract_atomtypes(ligand_itp)

    output_dir = os.path.dirname(os.path.abspath(output_top))
    itp_basename = os.path.basename(ligand_itp)
    dest_itp = os.path.join(output_dir, itp_basename)

    # Copy ITP file, removing [ atomtypes ] section to avoid duplication
    if os.path.abspath(ligand_itp) != os.path.abspath(dest_itp):
        import shutil
        _copy_itp_without_atomtypes(ligand_itp, dest_itp)
    else:
        # If in same location, create temp and move
        _copy_itp_without_atomtypes(ligand_itp, dest_itp + '.tmp')
        import shutil
        shutil.move(dest_itp + '.tmp', dest_itp)

    with open(protein_top) as fh:
        lines = fh.readlines()

    solvent_names = {'SOL', 'HOH', 'WAT', 'TIP3', 'NA', 'CL', 'K',
                     'Na', 'Cl', 'SOD', 'CLA', 'SOL2pos', 'SOL2neg'}

    with open(output_top, 'w') as ofh:
        atomtypes_inserted = False
        itp_inserted = False
        in_molecules = False
        ligand_molecule_inserted = False

        for i, line in enumerate(lines):
            stripped = line.split(';')[0].strip()
            tokens = stripped.split()

            # Insert atomtypes before first moleculetype if needed
            if atomtypes_lines and not atomtypes_inserted:
                if tokens == ['[', 'moleculetype', ']']:
                    ofh.write('[ atomtypes ]\n')
                    for at_line in atomtypes_lines:
                        ofh.write(at_line)
                    ofh.write('\n')
                    atomtypes_inserted = True

            # Insert #include for ligand ITP before [ system ]
            if tokens == ['[', 'system', ']'] and not itp_inserted:
                ofh.write(f'; Include ligand topology\n')
                ofh.write(f'#include "{itp_basename}"\n\n')
                itp_inserted = True

            # In [ molecules ] section, insert ligand before solvent
            if in_molecules and not ligand_molecule_inserted and tokens:
                if tokens[0] in solvent_names:
                    ofh.write(f'{ligand_name:<20s} 1\n')
                    ligand_molecule_inserted = True

            ofh.write(line)

            if tokens == ['[', 'molecules', ']']:
                in_molecules = True

        # If no solvent found, append at end
        if in_molecules and not ligand_molecule_inserted:
            ofh.write(f'{ligand_name:<20s} 1\n')


def _read_pdb_atoms(pdb_path: str):
    """Read ATOM/HETATM lines from PDB file."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(('ATOM  ', 'HETATM')):
                atoms.append(line.rstrip('\n'))
    return atoms


def _read_gro_atoms(gro_path: str):
    """Read atoms from GRO file and convert to PDB format."""
    atoms = []
    with open(gro_path) as fh:
        title = fh.readline()
        natoms = int(fh.readline().strip())
        for i in range(natoms):
            line = fh.readline()
            pdb_line = _gro_line_to_pdb(line, i + 1)
            atoms.append(pdb_line)
    return atoms


def _read_gro_as_pdb_atoms(gro_path: str):
    """Read ligand GRO file and convert atoms to PDB ATOM format.

    Note: Using ATOM instead of HETATM ensures compatibility with GROMACS
    tools like editconf which may skip HETATM records by default.
    """
    atoms = []
    with open(gro_path) as fh:
        title = fh.readline()
        natoms = int(fh.readline().strip())
        for i in range(natoms):
            line = fh.readline()
            pdb_line = _gro_line_to_pdb(line, i + 1, record='ATOM  ')
            atoms.append(pdb_line)
    return atoms


def _gro_line_to_pdb(gro_line: str, serial: int, record: str = 'ATOM  ') -> str:
    """Convert a single GRO line to PDB format.

    GRO format (fixed columns):
      Columns 1-5:   residue number (5 chars)
      Columns 6-10:  residue name (5 chars, left justified)
      Columns 11-15: atom name (5 chars, right justified)
      Columns 16-20: atom number (5 chars)
      Columns 21-28: x (nm, 8.3f)
      Columns 29-36: y (nm, 8.3f)
      Columns 37-44: z (nm, 8.3f)
    """
    resnum = int(gro_line[0:5].strip())
    resname = gro_line[5:10].strip()
    atomname = gro_line[10:15].strip()
    # atom_num = int(gro_line[15:20].strip())

    # Coordinates in nm -> convert to Angstroms for PDB
    x = float(gro_line[20:28]) * 10.0
    y = float(gro_line[28:36]) * 10.0
    z = float(gro_line[36:44]) * 10.0

    # Format atom name for PDB (4 chars, specific alignment rules)
    if len(atomname) < 4:
        atomname_fmt = f' {atomname:<3s}'
    else:
        atomname_fmt = f'{atomname:<4s}'

    # PDB format: ATOM serial atom resname chain resid x y z occ bfac
    # Note: Using space for chain ID (column 22)
    pdb_line = (f'{record}{serial:>5d} {atomname_fmt} {resname:>3s}  '
                f'{resnum:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00')
    return pdb_line


def _renumber_pdb_atom(line: str, serial: int) -> str:
    """Renumber the atom serial in a PDB ATOM/HETATM line."""
    if len(line) < 11:
        return line
    return f'{line[:6]}{serial:>5d}{line[11:]}'


def _extract_atomtypes(top_path: str):
    """Extract [ atomtypes ] entries from a GROMACS TOP file."""
    lines = []
    in_atomtypes = False
    with open(top_path) as fh:
        for line in fh:
            stripped = line.split(';')[0].strip()
            tokens = stripped.split()
            if tokens == ['[', 'atomtypes', ']']:
                in_atomtypes = True
                continue
            if in_atomtypes:
                if stripped.startswith('['):
                    break
                if stripped:
                    lines.append(line)
    return lines


def _copy_itp_without_atomtypes(source_itp: str, dest_itp: str):
    """Copy ITP file, removing [ atomtypes ] section.

    This prevents "Invalid order for directive" errors when including
    the ITP in a topology that already has atomtypes defined.
    """
    in_atomtypes = False
    with open(source_itp) as fh, open(dest_itp, 'w') as ofh:
        for line in fh:
            stripped = line.split(';')[0].strip()
            tokens = stripped.split()
            if tokens == ['[', 'atomtypes', ']']:
                in_atomtypes = True
                continue
            if in_atomtypes:
                if stripped.startswith('['):
                    # Reached next section, stop skipping
                    in_atomtypes = False
                    ofh.write(line)
                # Skip lines in atomtypes section
                continue
            ofh.write(line)
