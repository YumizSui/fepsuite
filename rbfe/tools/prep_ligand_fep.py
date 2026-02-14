#!/usr/bin/env python3
"""Preparation script for ligand relative binding free energy (RBFE) calculations.

This is the main prep script for the RBFE module. It:
1. Reads ligand SDF (multi-molecule) and protein PDB
2. Parameterizes each ligand with ACPYPE
3. Builds perturbation network (from user file or automatically)
4. For each edge: atom mapping, alignment, dual-topology merge
5. Combines with protein, solvates, adds ions
6. Creates run directories with para_conf.zsh
"""

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import rdkit.Chem

# Add feprest/tools and rbfe/tools to path
_script_dir = os.path.dirname(os.path.realpath(__file__))
_fepsuite_root = os.path.realpath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, os.path.join(_fepsuite_root, 'feprest', 'tools'))
sys.path.insert(0, _script_dir)

from ligand_param import parameterize_ligand
from combine_ligand import combine_structure, combine_topology_inline
from atom_mapping import find_mcs, get_atom_map, align_ligand_to_reference
from merge_ligand_top import merge_ligand_topologies
from perturbation_network import read_links_file, generate_optimal_network


def log_with_color(message):
    col = 33
    prefix = ">> "
    suffix = ""
    if sys.stdout.isatty():
        prefix = f"\x1b[{col};1m>> "
        suffix = "\x1b[0m"
    print(prefix + message + suffix)


def check_call_verbose(cmdline):
    log_with_color(" ".join(cmdline))
    subprocess.check_call(cmdline)


def load_ligands_from_sdf(sdf_path: str) -> Dict[str, rdkit.Chem.Mol]:
    """Load all molecules from a multi-molecule SDF file.

    Returns dict of {name: mol} with 3D coordinates and hydrogens.
    """
    suppl = rdkit.Chem.SDMolSupplier(sdf_path, removeHs=False)
    molecules = {}
    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"Warning: Failed to read molecule {i} from {sdf_path}")
            continue
        try:
            name = mol.GetProp('_Name')
        except KeyError:
            name = f"LIG{i}"
            mol.SetProp('_Name', name)
        if not name or name.strip() == '':
            name = f"LIG{i}"
            mol.SetProp('_Name', name)
        # Sanitize name (remove special characters)
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        mol.SetProp('_Name', safe_name)
        molecules[safe_name] = mol
    print(f"Loaded {len(molecules)} ligands from {sdf_path}")
    return molecules


def save_individual_sdfs(molecules: Dict[str, rdkit.Chem.Mol],
                         output_dir: str) -> Dict[str, str]:
    """Save individual SDF files for each ligand.

    Returns dict of {name: sdf_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    sdf_paths = {}
    for name, mol in molecules.items():
        sdf_path = os.path.join(output_dir, f"{name}.sdf")
        writer = rdkit.Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        sdf_paths[name] = sdf_path
    return sdf_paths


def parameterize_all_ligands(sdf_paths: Dict[str, str],
                             output_base_dir: str,
                             acpype_exe: str = "acpype") -> Dict[str, dict]:
    """Parameterize all ligands using ACPYPE.

    Returns dict of {name: ligand_info_dict}.
    """
    ligand_params = {}
    for name, sdf_path in sdf_paths.items():
        output_dir = os.path.join(output_base_dir, name)
        print(f"Parameterizing ligand: {name}")
        try:
            info = parameterize_ligand(sdf_path, output_dir,
                                       acpype_exe=acpype_exe)
            ligand_params[name] = info
        except Exception as e:
            print(f"Error parameterizing {name}: {e}")
            raise
    return ligand_params


def prepare_protein(protein_pdb: str, gmx: str, ff: str,
                    water_model: str, fepsuite: str, python3: str) -> dict:
    """Prepare protein topology using pdb2gmx.

    Returns dict with 'pdb', 'top', 'top_can' paths.
    """
    protein_dir = "protein"
    os.makedirs(protein_dir, exist_ok=True)

    pdb_out = os.path.join(protein_dir, "conf.pdb")
    top_out = os.path.join(protein_dir, "topol.top")

    if os.path.exists(pdb_out) and os.path.exists(top_out):
        print("Protein topology already exists, skipping pdb2gmx")
    else:
        # Strip HETATM from protein PDB
        protein_only = os.path.join(protein_dir, "protein_only.pdb")
        with open(protein_pdb) as fh, open(protein_only, 'w') as ofh:
            for line in fh:
                if not line.startswith('HETATM'):
                    ofh.write(line)

        curdir = os.getcwd()
        os.chdir(protein_dir)

        if os.path.exists(f"../{ff}.ff") and not os.path.exists(f"{ff}.ff"):
            os.symlink(f"../{ff}.ff", f"{ff}.ff")

        check_call_verbose(
            [gmx] + f"pdb2gmx -f protein_only.pdb -o conf.pdb -water {water_model} -ignh -ff {ff} -merge all".split()
        )

        os.chdir(curdir)

    # Canonicalize topology
    top_can = os.path.join(protein_dir, "topol_can.top")
    if not os.path.exists(top_can):
        import re
        curdir = os.getcwd()
        os.chdir(protein_dir)

        with open("topol.top") as ifh, open("topol_m.top", "w") as ofh:
            out = True
            for l in ifh:
                if re.search("POSRES", l):
                    out = False
                if out:
                    ofh.write(l)
                if re.search("ions.itp", l):
                    out = True

        check_call_verbose([python3, f"{fepsuite}/feprest/tools/pp.py",
                            "-o", "topol_pp.top", "topol_m.top"])
        check_call_verbose([python3, f"{fepsuite}/feprest/rest2py/canonicalize_top.py",
                            "topol_pp.top", "topol_can.top"])
        os.chdir(curdir)

    return {
        'pdb': os.path.abspath(pdb_out),
        'top': os.path.abspath(os.path.join(protein_dir, "topol.top")),
        'top_can': os.path.abspath(top_can),
    }


def prepare_edge(name_a: str, name_b: str,
                 molecules: Dict[str, rdkit.Chem.Mol],
                 ligand_params: Dict[str, dict],
                 protein_info: dict,
                 gmx: str, ff: str, water_model: str,
                 solv: float, ion: float,
                 ion_positive: str, ion_negative: str,
                 fepsuite: str, python3: str):
    """Prepare one edge (ligA -> ligB) of the perturbation network.

    Creates a directory with dual-topology ligand + protein, solvated.
    """
    edge_dir = f"{name_a}_to_{name_b}"
    log_with_color(f"Preparing edge: {name_a} -> {name_b}")

    os.makedirs(edge_dir, exist_ok=True)

    mol_a = molecules[name_a]
    mol_b = molecules[name_b]
    params_a = ligand_params[name_a]
    params_b = ligand_params[name_b]

    # 1. Create dual-topology ligand
    merged_itp = os.path.join(edge_dir, "merged_ligand.itp")
    merged_gro = os.path.join(edge_dir, "merged_ligand.gro")

    if not os.path.exists(merged_itp):
        merge_result = merge_ligand_topologies(
            mol_a, mol_b,
            params_a['itp'], params_b['itp'],
            merged_itp, merged_gro,
        )
        print(f"  Merged topology: {len(merge_result['atom_map'])} common atoms, "
              f"{len(merge_result['only_in_a'])} A-only, "
              f"{len(merge_result['only_in_b'])} B-only")
    else:
        print(f"  Merged topology already exists: {merged_itp}")

    # 2. Combine with protein
    curdir = os.getcwd()
    os.chdir(edge_dir)

    if os.path.exists(f"../{ff}.ff") and not os.path.exists(f"{ff}.ff"):
        os.symlink(f"../{ff}.ff", f"{ff}.ff")

    # Copy protein topology and structure
    protein_pdb = protein_info['pdb']
    protein_top = protein_info['top_can']

    if not os.path.exists("fepbase.pdb"):
        # Combine protein + ligand structures
        combine_structure(protein_pdb, "merged_ligand.gro", "fepbase.pdb")

    if not os.path.exists("fepbase.top"):
        # Get merged ligand name from ITP
        merged_name = _get_moleculetype_name(merged_itp)
        # Combine topologies
        combine_topology_inline(protein_top, merged_itp, merged_name,
                                "fepbase.top")

    # 3. Solvate
    if not os.path.exists("conf_ionized.pdb"):
        _solvate(gmx, ff, water_model, solv, ion, ion_positive, ion_negative)

    os.chdir(curdir)

    # Also prepare solvent-only (reference) system
    ref_dir = f"{edge_dir}_ref"
    os.makedirs(ref_dir, exist_ok=True)
    os.chdir(ref_dir)

    if os.path.exists(f"../{ff}.ff") and not os.path.exists(f"{ff}.ff"):
        os.symlink(f"../{ff}.ff", f"{ff}.ff")

    if not os.path.exists("fepbase.pdb"):
        # Reference: ligand only (no protein)
        shutil.copy(os.path.join("..", edge_dir, "merged_ligand.gro"), "merged_ligand.gro")
        shutil.copy(os.path.join("..", edge_dir, "merged_ligand.itp"), "merged_ligand.itp")

        # Create a minimal topology with just the ligand
        merged_name = _get_moleculetype_name("merged_ligand.itp")
        _write_ligand_only_top("fepbase.top", merged_name, "merged_ligand.itp", ff)

        # Convert GRO to PDB for solvation
        _gro_to_pdb("merged_ligand.gro", "fepbase.pdb")

    if not os.path.exists("conf_ionized.pdb"):
        ref_solv = 1.2  # larger box for reference
        _solvate(gmx, ff, water_model, ref_solv, ion, ion_positive, ion_negative)

    os.chdir(curdir)

    # Write para_conf.zsh for each edge
    _write_para_conf(edge_dir)
    _write_para_conf(ref_dir)

    return edge_dir, ref_dir


def _get_moleculetype_name(itp_path: str) -> str:
    """Extract moleculetype name from ITP."""
    in_moleculetype = False
    with open(itp_path) as fh:
        for line in fh:
            stripped = line.split(';')[0].strip()
            if not stripped:
                continue
            if 'moleculetype' in stripped and '[' in stripped:
                in_moleculetype = True
                continue
            if stripped.startswith('['):
                in_moleculetype = False
                continue
            if in_moleculetype:
                parts = stripped.split()
                if parts:
                    return parts[0]
    raise ValueError(f"Could not find moleculetype in {itp_path}")


def _solvate(gmx, ff, water_model, solv, ion, ion_positive, ion_negative):
    """Solvate and ionize the system in the current directory."""
    check_call_verbose(
        [gmx] + f"editconf -f fepbase.pdb -d {solv} -bt dodecahedron -o conf_box.pdb".split()
    )

    maybe_relative = ""
    if os.path.exists(f"{ff}.ff"):
        maybe_relative = "./"

    with open("fepbase.top") as fh, open("topol_solvated.top", "w") as ofh:
        for l in fh:
            ls = l.split()
            if ls == ['[', 'system', ']']:
                print(f'#include "{maybe_relative}{ff}.ff/{water_model}.itp"', file=ofh)
                print(f'#include "{maybe_relative}{ff}.ff/ions.itp"', file=ofh)
            ofh.write(l)

    waterbox = "spc216.gro"
    if water_model in ["tip4p", "tip4pew"]:
        waterbox = "tip4p.gro"

    check_call_verbose(
        [gmx] + f"solvate -cp conf_box.pdb -p topol_solvated -cs {waterbox} -o conf_solvated.pdb".split()
    )

    with open("dummy.mdp", "w") as ofh:
        pass

    check_call_verbose(
        [gmx] + "grompp -f dummy.mdp -p topol_solvated.top -c conf_solvated.pdb -po dummy_out -o topol_solvated -maxwarn 1".split()
    )

    shutil.copy("topol_solvated.top", "topol_ionized.top")
    cmds = [gmx] + f"genion -s topol_solvated -o conf_ionized.pdb -p topol_ionized.top -pname {ion_positive} -nname {ion_negative} -conc {ion} -neutral".split()
    log_with_color(" ".join(cmds))
    proc = subprocess.Popen(cmds, stdin=subprocess.PIPE)
    proc.communicate(b"SOL\n")


def _write_ligand_only_top(top_path, lig_name, lig_itp, ff):
    """Write a minimal topology for ligand-only (reference) system."""
    with open(top_path, 'w') as ofh:
        ofh.write('; Ligand-only topology for RBFE reference calculation\n')
        ofh.write(f'#include "{ff}.ff/forcefield.itp"\n\n')
        ofh.write(f'#include "{lig_itp}"\n\n')
        ofh.write('[ system ]\n')
        ofh.write(f'Ligand {lig_name}\n\n')
        ofh.write('[ molecules ]\n')
        ofh.write(f'{lig_name}    1\n')


def _gro_to_pdb(gro_path, pdb_path):
    """Simple GRO to PDB conversion."""
    from combine_ligand import _read_gro_as_pdb_atoms, _renumber_pdb_atom
    atoms = _read_gro_as_pdb_atoms(gro_path)
    with open(pdb_path, 'w') as ofh:
        for i, atom in enumerate(atoms):
            ofh.write(_renumber_pdb_atom(atom, i + 1) + '\n')
        ofh.write('END\n')


def _write_para_conf(edge_dir):
    """Write para_conf.zsh for an edge directory."""
    para_conf = os.path.join(edge_dir, "para_conf.zsh")
    if not os.path.exists(para_conf):
        with open(para_conf, 'w') as ofh:
            ofh.write("# RBFE edge parameters\n")
            ofh.write("# Automatically generated by prep_ligand_fep.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ligand RBFE calculations",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    gmxbin = None
    if "GMXBIN" in os.environ:
        gmxbin = os.environ["GMXBIN"] + "/gmx"

    fepsuitepath = os.path.realpath(os.path.join(_script_dir, '..', '..'))

    parser.add_argument("--protein", required=True, help="Protein PDB file")
    parser.add_argument("--ligands", required=True, help="Multi-molecule SDF file with all ligands")
    parser.add_argument("--links", default=None,
                        help="Perturbation links file (ligA ligB per line)")
    parser.add_argument("--auto-network", action="store_true",
                        help="Automatically generate perturbation network")
    parser.add_argument("--gmx", default=gmxbin,
                        required=(gmxbin is None), help="Path to GROMACS gmx")
    parser.add_argument("--ff", required=True, help="Force field name")
    parser.add_argument("--water-model", default="tip3p", help="Water model")
    parser.add_argument("--solv", default=0.5, type=float,
                        help="Water thickness (nm)")
    parser.add_argument("--ion", default=0.15, type=float,
                        help="Ion concentration (mol/L)")
    parser.add_argument("--ion-positive", default="NA", help="Positive ion name")
    parser.add_argument("--ion-negative", default="CL", help="Negative ion name")
    parser.add_argument("--acpype", default="acpype", help="ACPYPE executable path")
    parser.add_argument("--fepsuite", default=fepsuitepath, help="FEPsuite root")
    parser.add_argument("--python3", default=sys.executable, help="Python3 path")

    args = parser.parse_args()

    # 1. Load ligands
    molecules = load_ligands_from_sdf(args.ligands)

    # 2. Save individual SDFs
    sdf_paths = save_individual_sdfs(molecules, "ligand_sdfs")

    # 3. Parameterize all ligands
    ligand_params = parameterize_all_ligands(sdf_paths, "ligand_params",
                                             acpype_exe=args.acpype)

    # 4. Get perturbation network
    if args.links:
        pairs = read_links_file(args.links)
        # Validate pair names
        for a, b in pairs:
            if a not in molecules:
                raise ValueError(f"Ligand '{a}' in links file not found in SDF")
            if b not in molecules:
                raise ValueError(f"Ligand '{b}' in links file not found in SDF")
    elif args.auto_network:
        pairs = generate_optimal_network(molecules)
    else:
        raise ValueError("Either --links or --auto-network must be specified")

    print(f"\nPerturbation network: {len(pairs)} edges")
    for a, b in pairs:
        print(f"  {a} -> {b}")

    # 5. Prepare protein
    protein_info = prepare_protein(args.protein, args.gmx, args.ff,
                                   args.water_model, args.fepsuite, args.python3)

    # 6. Prepare each edge
    edge_dirs = []
    ref_dirs = []
    for name_a, name_b in pairs:
        edge_dir, ref_dir = prepare_edge(
            name_a, name_b, molecules, ligand_params, protein_info,
            args.gmx, args.ff, args.water_model,
            args.solv, args.ion, args.ion_positive, args.ion_negative,
            args.fepsuite, args.python3,
        )
        edge_dirs.append(edge_dir)
        ref_dirs.append(ref_dir)

    # 7. Summary
    print(f"\nRBFE preparation complete!")
    print(f"Complex (holo) directories:")
    for d in edge_dirs:
        print(f"  {d}")
    print(f"Reference (solvent) directories:")
    for d in ref_dirs:
        print(f"  {d}")

    # Write edges summary file
    with open("edges.txt", 'w') as ofh:
        for (a, b), edge_d, ref_d in zip(pairs, edge_dirs, ref_dirs):
            ofh.write(f"{a}\t{b}\t{edge_d}\t{ref_d}\n")
    print(f"\nEdge summary written to edges.txt")


if __name__ == "__main__":
    main()
