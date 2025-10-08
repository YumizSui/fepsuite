#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import warnings
from typing import Dict, List, Tuple, Sequence

from prep_mutation_fep import PDBInfoSummary, update_mutinfo, parse_pdb

from mutation import Mutation, parse_mutations

def write_faspr_seq(ddir: str, basepdbinfo: PDBInfoSummary, mutations: Sequence[Mutation]):
    """Generate FASPR seq.txt file."""
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "seq.txt"), "w") as ofh:
        curchain = None
        reschainmap = basepdbinfo.res_chain_dic
        totmut = 0
        for (c, r) in basepdbinfo.order:
            if curchain != c:
                if curchain is not None:
                    print("", file=ofh)
                chainname = c if c.strip() != "" else "X"
                print(f">{chainname}", file=ofh)
                curchain = c
            res = reschainmap[r][c]
            mutated = False
            for m in mutations:
                if m.resid == r and (m.chain is None or m.chain == c):
                    if m.before_res is not None and m.before_res != res:
                        raise RuntimeError(f"Mutation ({m}) is not compatible with chain '{c}' resid {r}")
                    res = m.after_res
                    mutated = True
                    totmut += 1
                    break
            if (res == "C") and (not mutated):
                res = "c"
            print(res, end="", file=ofh)
        print("", file=ofh)
        if totmut != len(mutations):
            raise RuntimeError(f"Not all mutations are consumed, requested: {mutations}")


def main():
    p = argparse.ArgumentParser(description="Generate mutated PDB from WT PDB with specified mutations")
    p.add_argument("--pdb", required=True, help="Input WT base PDB file")
    p.add_argument("--mut", default="", help="Mutation string (e.g., H:35A_H:50V). If empty, generates repacked WT")
    p.add_argument("--faspr", required=True, help="Path to FASPR binary")
    p.add_argument("--out", required=True, help="Output PDB file")
    p.add_argument("--workdir", default="mutated_build", help="Working directory for intermediate files")
    p.add_argument("--seed", type=int, default=None, help="Random seed for FASPR (optional)")
    args = p.parse_args()

    base = parse_pdb(args.pdb)
    if args.mut.strip():
        muts = parse_mutations(args.mut)
        muts = update_mutinfo(muts, base)
    else:
        muts = []

    write_faspr_seq(args.workdir, base, muts)

    cmd = [args.faspr, "-i", args.pdb, "-o", os.path.join(args.workdir, "completed.pdb"), "-s", os.path.join(args.workdir, "seq.txt")]
    if args.seed is not None:
        cmd += ["-seed", str(args.seed)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    out_tmp = os.path.join(args.workdir, "completed.pdb")
    if (not os.path.exists(out_tmp)) or (os.path.getmtime(out_tmp) < os.path.getmtime(os.path.join(args.workdir, "seq.txt"))):
        raise RuntimeError("FASPR run failed")
    shutil.copy2(out_tmp, args.out)
    print(f"Done. Wrote mutated PDB to: {args.out}")

if __name__ == "__main__":
    main()
