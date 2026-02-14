#!/usr/bin/env python3
"""Ligand parameterization module using ACPYPE.

Simplified port of PyAutoFEP mol_util.py:parameterize_small_molecule().
Converts SDF ligands to GROMACS ITP/GRO via ACPYPE with GAFF2 + BCC charges.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional


def parameterize_ligand(sdf_path: str, output_dir: str,
                        acpype_exe: str = "acpype",
                        atom_type: str = "gaff2",
                        charge_method: str = "bcc",
                        net_charge: Optional[int] = None,
                        timeout: Optional[int] = None) -> dict:
    """Parameterize a ligand from SDF using ACPYPE.

    Parameters
    ----------
    sdf_path : str
        Path to input SDF file (single molecule).
    output_dir : str
        Directory to write output files.
    acpype_exe : str
        Path to ACPYPE executable.
    atom_type : str
        Atom type for ACPYPE ('gaff', 'gaff2', 'amber', 'amber2').
    charge_method : str
        Charge method for ACPYPE ('bcc', 'gas', 'user').
    net_charge : int or None
        Net formal charge. If None, computed from SDF via RDKit.
    timeout : int or None
        Timeout in seconds for ACPYPE run.

    Returns
    -------
    dict
        {'itp': path, 'gro': path, 'top': path, 'posre': path, 'ligand_name': name}
    """
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    # Determine ligand name from filename
    ligand_name = os.path.splitext(os.path.basename(sdf_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    # Check for cached results
    itp_path = os.path.join(output_dir, f"{ligand_name}_GMX.itp")
    top_path = os.path.join(output_dir, f"{ligand_name}_GMX.top")
    gro_path = os.path.join(output_dir, f"{ligand_name}_GMX.gro")
    posre_path = os.path.join(output_dir, f"posre_{ligand_name}.itp")

    if all(os.path.isfile(f) for f in [itp_path, top_path, posre_path]):
        print(f"Ligand parameters already exist in {output_dir}, skipping ACPYPE")
        return {
            'itp': itp_path,
            'top': top_path,
            'gro': gro_path,
            'posre': posre_path,
            'ligand_name': ligand_name,
        }

    # Compute net charge from SDF if not given
    if net_charge is None:
        net_charge = _compute_formal_charge(sdf_path)

    # Build ACPYPE command line
    cmd = [acpype_exe, '-i', sdf_path, '-n', str(net_charge),
           '-o', 'gmx', '-a', atom_type, '-c', charge_method,
           '-b', ligand_name]

    # Set OMP_NUM_THREADS=1 for sqm efficiency
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    print(f"Running ACPYPE: {' '.join(cmd)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy SDF to temp directory
        tmp_sdf = os.path.join(tmpdir, os.path.basename(sdf_path))
        shutil.copy2(sdf_path, tmp_sdf)
        # Update command to use local file
        cmd[2] = os.path.basename(sdf_path)

        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=tmpdir, timeout=timeout, env=env)

        if result.returncode != 0:
            raise RuntimeError(
                f"ACPYPE failed (return code {result.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Copy result files from ACPYPE output directory
        acpype_dir = os.path.join(tmpdir, f"{ligand_name}.acpype")
        if not os.path.isdir(acpype_dir):
            raise RuntimeError(
                f"ACPYPE output directory not found: {acpype_dir}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        file_map = {
            'itp': (f"{ligand_name}_GMX.itp", itp_path),
            'top': (f"{ligand_name}_GMX.top", top_path),
            'gro': (f"{ligand_name}_GMX.gro", gro_path),
            'posre': (f"posre_{ligand_name}.itp", posre_path),
        }

        for key, (src_name, dst_path) in file_map.items():
            src = os.path.join(acpype_dir, src_name)
            if os.path.isfile(src):
                shutil.copy2(src, dst_path)
            elif key in ('itp', 'top'):
                raise FileNotFoundError(
                    f"Expected ACPYPE output file not found: {src}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

    return {
        'itp': itp_path,
        'top': top_path,
        'gro': gro_path if os.path.isfile(gro_path) else None,
        'posre': posre_path if os.path.isfile(posre_path) else None,
        'ligand_name': ligand_name,
    }


def use_existing_itp(itp_path: str, gro_path: str,
                     ligand_name: Optional[str] = None) -> dict:
    """Use pre-prepared ITP + GRO files for a ligand.

    Parameters
    ----------
    itp_path : str
        Path to the ligand ITP file.
    gro_path : str
        Path to the ligand GRO coordinate file.
    ligand_name : str or None
        Residue/molecule name. If None, extracted from ITP.

    Returns
    -------
    dict
        {'itp': path, 'gro': path, 'top': None, 'posre': None, 'ligand_name': name}
    """
    if not os.path.isfile(itp_path):
        raise FileNotFoundError(f"ITP file not found: {itp_path}")
    if not os.path.isfile(gro_path):
        raise FileNotFoundError(f"GRO file not found: {gro_path}")

    if ligand_name is None:
        ligand_name = _extract_moleculetype_name(itp_path)

    return {
        'itp': os.path.abspath(itp_path),
        'gro': os.path.abspath(gro_path),
        'top': None,
        'posre': None,
        'ligand_name': ligand_name,
    }


def _compute_formal_charge(sdf_path: str) -> int:
    """Compute formal charge from SDF file using RDKit."""
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next(suppl)
        if mol is None:
            raise ValueError(f"Failed to read molecule from {sdf_path}")
        return Chem.GetFormalCharge(mol)
    except ImportError:
        print("Warning: RDKit not available, assuming net charge = 0")
        return 0


def _extract_moleculetype_name(itp_path: str) -> str:
    """Extract molecule name from [ moleculetype ] section of ITP."""
    in_moleculetype = False
    with open(itp_path) as fh:
        for line in fh:
            stripped = line.split(';')[0].strip()
            if not stripped:
                continue
            if '[ moleculetype ]' in line or (stripped.startswith('[') and 'moleculetype' in stripped):
                in_moleculetype = True
                continue
            if stripped.startswith('['):
                in_moleculetype = False
                continue
            if in_moleculetype:
                parts = stripped.split()
                if parts:
                    return parts[0]
    raise ValueError(f"Could not find moleculetype name in {itp_path}")
