import os
import subprocess

import itertools
import networkx as nx
import shutil
import argparse
import pickle
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FEP calculations")
    parser.add_argument("--partners", nargs="*", default=["partner1"], help="List of partners")
    parser.add_argument("--target", required=True, help="Target identifier (e.g., 1AO7_ABC_DE)")
    parser.add_argument("--base-target-dir", required=True, help="Base target directory path")
    parser.add_argument("--graph-file", default="Graph.pkl", help="Path to graph pickle file")

    args = parser.parse_args()

    partners = args.partners
    target = args.target
    partner_chains_dict = {"partner1": list(target.split("_")[1]), "partner2": list(target.split("_")[2])}
    base_target_dir = Path(args.base_target_dir)
    with open(args.graph_file, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.DiGraph)

    FEPSUITE_DIR = "/home/6/uc02086/workspace-kf/fep/fepsuite"
    calculation_files = []
    failed_files = []
    for u, v in G.edges():
        for mode in partners:
            input_pdb = Path(G.nodes[u][f"{mode}_pdb_file"])
            mut_v = v.split("_")
            mut_u = u.split("_")
            new_mut_v = [m for m in mut_v if m not in mut_u]
            new_mut_chains = set([m[0] for m in new_mut_v])
            if mode.startswith("partner"):
                is_hit = False
                for chain in new_mut_chains:
                    if chain in partner_chains_dict[mode]:
                        is_hit = True
                        break
                if not is_hit:
                    continue
            mutation_v = "_".join(new_mut_v)
            target_dir = base_target_dir/mode/u
            if (target_dir / f"{mutation_v}.done").exists():
                calculation_files.append(str(target_dir / f"wt_{mutation_v}"))
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            ff_link = target_dir / "amber14sb_OL15_fs1.ff"
            if not ff_link.exists():
                ff_source = Path(FEPSUITE_DIR) / "forcefields" / "amber14sb_OL15_fs1.ff"
                ff_link.symlink_to(ff_source)
            ret = subprocess.run([
                "python3",
                str(Path(FEPSUITE_DIR) / "feprest" / "tools" / "prep_mutation_fep.py"),
                "--faspr",
                str(Path(FEPSUITE_DIR) / "feprest" / "FASPR" / "FASPR"),
                "--pdb", str(input_pdb),
                "--mutation", mutation_v,
                "--ff", "amber14sb_OL15_fs1"
            ], cwd=str(target_dir))
            if ret.returncode != 0:
                print(f"Error: {ret.returncode}")
                print(ret.stderr)
                print(ret.stdout)
                failed_files.append(str(target_dir / f"wt_{mutation_v}"))
            else:
                # touch target_dir/v.done
                (target_dir / f"{mutation_v}.done").touch()
                calculation_files.append(str(target_dir / f"wt_{mutation_v}"))
    print(f"calculation_files ({len(calculation_files)}):")
    for file in calculation_files:
        print(file)
    print(f"failed_files ({len(failed_files)}):")
    for file in failed_files:
        print(file)
