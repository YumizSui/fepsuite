#!/usr/bin/env python3
"""
Clean results script

Usage:
    python clean_fepsuite_results.py --csv <file> [--ref-only] [--dry-run] [--nproc N]
"""

import argparse
import csv
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def worker_clean_workdir(args_tuple: Tuple[str, bool, bool, bool, str]) -> Tuple[bool, str, Optional[str]]:
    """
    Worker function for multiprocessing.
    Returns: (success, workdir_str, error_message)
    """
    workdir_str, ref_only, dry_run, force, current_dir_str = args_tuple

    # Create a temporary cleaner instance for this worker
    cleaner = ResultsCleaner(dry_run=dry_run, force=force)
    cleaner.current_dir = Path(current_dir_str)

    try:
        success = cleaner.clean_workdir(workdir_str, ref_only)
        return (success, workdir_str, None)
    except Exception as e:
        return (False, workdir_str, str(e))


class ResultsCleaner:
    """Class for cleaning up FEP calculation results"""

    def __init__(self, dry_run: bool = False, force: bool = False):
        # Use the directory where the script is located as the base directory
        self.current_dir = Path(__file__).parent.resolve()
        self.dry_run = dry_run
        self.force = force
        self.failed_dirs = []

    def log(self, message: str, level: str = "INFO"):
        """Log output"""
        print(f"[{level}] {message}")

    def read_csv_target_dirs(self, csv_file: str) -> List[str]:
        """Read list of work directories from target_dir column in CSV file"""
        try:
            workdirs = []

            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                if 'target_dir' not in reader.fieldnames:
                    self.log(f"Error: 'target_dir' column not found in {csv_file}", "ERROR")
                    self.log(f"Available columns: {list(reader.fieldnames)}", "ERROR")
                    sys.exit(1)

                for row in reader:
                    target_dir = row.get('target_dir', '').strip()
                    if target_dir:  # Exclude empty strings and None
                        workdirs.append(target_dir)

            self.log(f"Read {len(workdirs)} work directories from {csv_file}")
            return workdirs

        except FileNotFoundError:
            self.log(f"CSV file not found: {csv_file}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log(f"Error reading CSV file: {e}", "ERROR")
            sys.exit(1)

    @staticmethod
    def get_bar1_dG(file_path: str) -> float:
        with open(file_path, "r") as f:
            x = f.read().splitlines()[-1]
            _, dg_val, dg_err = x.split()
        return float(dg_val), float(dg_err)

    def is_calculation_finished(self, workdir: Path) -> bool:
        """Check if calculation is finished (check existence of bar1.log or results/bar1.log)"""
        bar_log = workdir / "bar1.log"
        if not bar_log.exists():
            bar_log = workdir / "results" / "bar1.log"
        if not bar_log.exists():
            return False
        try:
            dg_val, dg_err = self.get_bar1_dG(bar_log)
        except:
            return False
        if dg_val is None or dg_err is None:
            return False
        return True

    def is_already_cleaned(self, workdir: Path) -> bool:
        """Check if already cleaned up"""
        cleaned_marker = workdir / "cleaned"
        return cleaned_marker.exists()

    def is_reference_dir(self, workdir: str) -> bool:
        """Determine if it's a reference directory"""
        return workdir.endswith("_ref")

    def generate_trajectory_files(self, workdir: Path) -> bool:
        """Generate trajectory files (execute run.zsh)"""
        try:
            os.chdir(self.current_dir)
            cmd = ["./run_ctrl.zsh", str(workdir), "999"]

            if self.dry_run:
                self.log(f"DRY RUN: Would execute: {' '.join(cmd)}")
                return True

            self.log(f"Generating trajectory files for {workdir}")
            self.log(" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.log(f"Failed to generate trajectory files: {result.stderr}", "ERROR")
                return False

            # Check existence of index file
            index_file = workdir / "prodrun" / "fepbase_A.ndx"
            if not index_file.exists():
                self.log(f"Index file not generated: {index_file}", "ERROR")
                return False

            return True

        except Exception as e:
            self.log(f"Error generating trajectory files: {e}", "ERROR")
            return False

    def copy_files_to_results(self, workdir: Path, is_ref: bool = False):
        """Copy necessary files to results directory"""
        results_dir = workdir / "results"

        if self.dry_run:
            self.log(f"DRY RUN: Would create results directory in {workdir}")
        else:
            results_dir.mkdir(exist_ok=True)

        # Copy basic files (bar1.log is not copied to keep it in original location)
        basic_files = ["fepbase.pdb", "fepbase.top", "conf_ionized.pdb", "topol_ionized.top"]

        for filename in basic_files:
            src = workdir / filename
            dst = results_dir / filename

            if src.exists():
                if self.dry_run:
                    self.log(f"DRY RUN: Would copy {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)
                    self.log(f"Copied {filename}")
            else:
                self.log(f"Warning: {filename} not found in {workdir}", "WARN")

        # Additional files for non-reference directories
        if not is_ref:
            additional_files = ["fepbase_A.pdb", "fepbase_B.pdb"]
            additional_files_prodrun = ["fepbase_A.ndx", "fepbase_B.ndx", "stateA.xtc"]

            for filename in additional_files:
                src = workdir / filename
                dst = results_dir / filename

                if src.exists():
                    if self.dry_run:
                        self.log(f"DRY RUN: Would copy {src} -> {dst}")
                    else:
                        shutil.copy2(src, dst)
                        self.log(f"Copied {filename}")

            prodrun_dir = workdir / "prodrun"
            for filename in additional_files_prodrun:
                src = prodrun_dir / filename
                dst = results_dir / filename

                if src.exists():
                    if self.dry_run:
                        self.log(f"DRY RUN: Would copy {src} -> {dst}")
                    else:
                        shutil.copy2(src, dst)
                        self.log(f"Copied {filename} from prodrun")

    def copy_deltae_files(self, workdir: Path):
        """Copy deltae files and compress reps directory with tar.gz"""
        results_reps_dir = workdir / "results" / "reps"

        if self.dry_run:
            self.log(f"DRY RUN: Would create {results_reps_dir}")
        else:
            results_reps_dir.mkdir(exist_ok=True)

        prodrun_dir = workdir / "prodrun"
        if not prodrun_dir.exists():
            self.log(f"Warning: prodrun directory not found in {workdir}", "WARN")
            return

        # Search for rep* directories
        rep_dirs = list(prodrun_dir.glob("rep*/"))

        for rep_dir in rep_dirs:
            if rep_dir.is_dir():
                rep_name = rep_dir.name
                src = rep_dir / "deltae.xvg"
                dst = results_reps_dir / f"deltae_{rep_name}.xvg"

                if src.exists():
                    if self.dry_run:
                        self.log(f"DRY RUN: Would copy {src} -> {dst}")
                    else:
                        shutil.copy2(src, dst)
                        self.log(f"Copied deltae_{rep_name}.xvg")
                else:
                    self.log(f"Warning: {src} not found", "WARN")

        # Compress reps directory with tar.gz
        if not self.dry_run and results_reps_dir.exists():
            results_dir = workdir / "results"
            tar_file = results_dir / "reps.tar.gz"

            try:
                # tar zcvf reps.tar.gz reps/
                cmd = ["tar", "zcvf", str(tar_file), "-C", str(results_dir), "reps"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=results_dir)

                if result.returncode == 0:
                    # After successful compression, delete the original reps directory
                    shutil.rmtree(results_reps_dir)
                    self.log(f"Compressed reps directory to reps.tar.gz")
                else:
                    self.log(f"Failed to compress reps directory: {result.stderr}", "WARN")
            except Exception as e:
                self.log(f"Error compressing reps directory: {e}", "WARN")
        elif self.dry_run:
            self.log(f"DRY RUN: Would compress {results_reps_dir} to tar.gz")

    def cleanup_workdir(self, workdir: Path):
        """Delete unnecessary files and cleanup"""
        if self.dry_run:
            self.log(f"DRY RUN: Would cleanup {workdir}")
            return

        # Delete everything except results directory
        for item in workdir.iterdir():
            if item.name != "results" and item.name != "bar1.log":  # Keep bar1.log
                if item.is_dir():
                    # Use unlink() for symbolic links
                    if item.is_symlink():
                        item.unlink()
                        self.log(f"Removed symbolic link: {item.name}")
                    else:
                        shutil.rmtree(item)
                        self.log(f"Removed directory: {item.name}")
                else:
                    item.unlink()
                    self.log(f"Removed file: {item.name}")

        # Move contents of results directory to parent level
        results_dir = workdir / "results"
        if results_dir.exists():
            for item in results_dir.iterdir():
                dst = workdir / item.name
                if item.is_dir():
                    shutil.move(str(item), str(dst))
                else:
                    shutil.move(str(item), str(dst))
                self.log(f"Moved {item.name} to root")

            # Remove empty results directory
            results_dir.rmdir()
            self.log("Removed empty results directory")

    def create_cleaned_marker(self, workdir: Path):
        """Create cleaned marker file"""
        if self.dry_run:
            self.log(f"DRY RUN: Would create cleaned marker in {workdir}")
        else:
            cleaned_marker = workdir / "cleaned"
            cleaned_marker.touch()
            self.log("Created cleaned marker")

    def write_failed_list(self, ref_mode: bool = False):
        """Output list of failed directories"""
        if not self.failed_dirs:
            return

        suffix = "_ref" if ref_mode else ""
        failed_file = self.current_dir / f"failed_cleaning{suffix}.txt"

        if self.dry_run:
            self.log(f"DRY RUN: Would write {len(self.failed_dirs)} failed directories to {failed_file}")
        else:
            with open(failed_file, 'w') as f:
                for failed_dir in self.failed_dirs:
                    f.write(f"{failed_dir}\n")
            self.log(f"Written {len(self.failed_dirs)} failed directories to {failed_file}")

    def clean_workdir(self, workdir_str: str, ref_only: bool = False) -> bool:
        """Clean up a single work directory"""
        workdir = Path(workdir_str).resolve()

        self.log(f"Processing: {workdir}")

        # Check if already cleaned up
        if not self.force and self.is_already_cleaned(workdir) and not ref_only:
            self.log(f"Already cleaned: {workdir}")
            return True

        # Check calculation completion
        if not self.is_calculation_finished(workdir):
            self.log(f"Calculation not finished yet: {workdir}", "WARN")
            self.failed_dirs.append(str(workdir))
            return False

        try:
            os.chdir(workdir)

            is_ref = self.is_reference_dir(workdir_str) or ref_only

            # For non-reference directories, generate trajectory files
            if not is_ref:
                if not self.generate_trajectory_files(workdir):
                    self.log(f"Failed to generate trajectory files: {workdir}", "ERROR")
                    self.failed_dirs.append(str(workdir))
                    return False

            # File copying
            self.copy_files_to_results(workdir, is_ref)
            self.copy_deltae_files(workdir)

            # Cleanup
            self.cleanup_workdir(workdir)

            # Create cleaned marker (not created in reference mode)
            if not ref_only:
                self.create_cleaned_marker(workdir)

            self.log(f"Successfully cleaned: {workdir}")
            return True

        except Exception as e:
            self.log(f"Error processing {workdir}: {e}", "ERROR")
            self.failed_dirs.append(str(workdir))
            return False

    def clean_all(self, csv_file: str, ref_only: bool = False, nproc: int = 1):
        """Execute cleanup for all work directories"""
        workdirs = self.read_csv_target_dirs(csv_file)

        success_count = 0
        total_count = len(workdirs)

        if nproc == 1:
            # Single process processing (traditional way)
            for workdir in workdirs:
                if self.clean_workdir(workdir, ref_only):
                    success_count += 1
        else:
            # Multi-process processing
            self.log(f"Using {nproc} processes for parallel processing")

            # Prepare arguments to pass to each worker
            worker_args = [
                (workdir, ref_only, self.dry_run, self.force, str(self.current_dir))
                for workdir in workdirs
            ]

            # Use multiprocessing pool
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.map(worker_clean_workdir, worker_args)

            # Process results
            for success, workdir_str, error_msg in results:
                if success:
                    success_count += 1
                    self.log(f"Successfully cleaned: {workdir_str}")
                else:
                    self.failed_dirs.append(workdir_str)
                    if error_msg:
                        self.log(f"Error processing {workdir_str}: {error_msg}", "ERROR")
                    else:
                        self.log(f"Failed to clean: {workdir_str}", "ERROR")

        # Output failed list
        self.write_failed_list(ref_only)

        self.log(f"Cleanup completed: {success_count}/{total_count} successful")

        if self.failed_dirs:
            self.log(f"Failed directories: {len(self.failed_dirs)}", "WARN")
            for failed_dir in self.failed_dirs:
                self.log(f"  - {failed_dir}", "WARN")


def main():
    parser = argparse.ArgumentParser(
        description="Clean FEP calculation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal cleanup
  python clean_fepsuite_results.py --csv mutations.csv

  # Parallel processing with 4 processes
  python clean_fepsuite_results.py --csv mutations.csv --nproc 4

  # Reference directories only
  python clean_fepsuite_results.py --csv mutations.csv --ref-only

  # Dry run (show what would be done without actually doing it)
  python clean_fepsuite_results.py --csv mutations.csv --dry-run

  # Force execution (skip calculation completion check)
  python clean_fepsuite_results.py --csv mutations.csv --force
        """
    )

    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="CSV file containing target_dir column with work directories"
    )

    parser.add_argument(
        "--ref-only", "-r",
        action="store_true",
        help="Process reference directories only (skip trajectory generation)"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without actually doing it"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force cleanup without checking if calculation is finished"
    )

    parser.add_argument(
        "--nproc", "-j",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing (default: 1)"
    )

    args = parser.parse_args()

    # Argument validation
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Argument validation (nproc)
    if args.nproc < 1:
        print(f"Error: nproc must be >= 1, got {args.nproc}")
        sys.exit(1)

    # Execute cleanup
    cleaner = ResultsCleaner(args.dry_run, args.force)
    cleaner.clean_all(args.csv, args.ref_only, args.nproc)


if __name__ == "__main__":
    main()
