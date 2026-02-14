#!/usr/bin/env python

"""
MBAR analysis script for FEP simulations.
Performs comprehensive analysis including overlap matrices, free energy curves,
effective sample sizes, and convergence diagnostics.
"""

import pymbar
import sys
import re
import argparse
import numpy as np
import os.path
import pickle
import collections
import json
import tarfile
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import parse_deltae from bar_deltae
from feprest.bar_deltae import parse_deltae

SimEval = collections.namedtuple("SimEval", ["sim", "eval"])

# in kJ/mol/K (to fit GROMACSy output)
gasconstant = 0.008314472
# in kcal/mol/K
gasconstant_kcal = 0.0019872036

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class MBARAnalysis:
    """
    MBAR analysis class for FEP simulations.

    This class provides comprehensive MBAR analysis including:
    - Overlap matrix computation and visualization
    - Free energy curve plotting
    - Effective sample size (ESS) analysis
    - Time-windowed convergence analysis
    - Summary statistics

    Example:
        >>> from mbar_analysis import MBARAnalysis
        >>> analyzer = MBARAnalysis(
        ...     xvgs="path/to/rep%sim/deltae.xvg",
        ...     nsim=32,
        ...     temp=300.0,
        ...     save_dir="./results"
        ... )
        >>> analyzer.run()  # Run full analysis
        >>> # Or run individual steps:
        >>> analyzer.load_data()
        >>> analyzer.build_matrix()
        >>> analyzer.compute_mbar()
        >>> analyzer.plot_all()
    """

    def __init__(self, xvgs, nsim, temp=300.0, save_dir=None, subsample=1,
                 minpart=None, maxpart=None, time_begin=None, time_end=None,
                 output_format='png', overlap_threshold=0.01, ess_threshold=0.1,
                 n_windows=10, verbose=False, tar_file=None):
        """
        Initialize MBAR analysis.

        Parameters:
        -----------
        xvgs : str
            xvg file path pattern, "%sim" will be replaced by simulation number
            If tar_file is specified, this should be the path pattern within the tar file
            (e.g., "reps/deltae_rep%sim.xvg")
        nsim : int
            Number of simulations (lambda windows)
        temp : float, default=300.0
            Temperature in Kelvin
        save_dir : str, default=None
            Directory to save results. If None, uses current directory.
        subsample : int, default=1
            Subsampling interval
        minpart : int, default=None
            Part number begin (for multi-part simulations)
        maxpart : int, default=None
            Part number end (for multi-part simulations)
        time_begin : float, default=None
            Beginning time in ps (for time filtering)
        time_end : float, default=None
            End time in ps (for time filtering)
        output_format : str, default='png'
            Output format for plots ('png', 'pdf', or 'svg')
        overlap_threshold : float, default=0.01
            Minimum overlap threshold for warnings
        ess_threshold : float, default=0.1
            Minimum ESS/N ratio for warnings
        n_windows : int, default=10
            Number of time windows for convergence analysis
        verbose : bool, default=True
            Print progress messages
        tar_file : str, default=None
            Path to tar.gz file containing xvg files. If specified, xvgs should be
            the path pattern within the tar file (e.g., "reps/deltae_rep%sim.xvg")
        """
        self.xvgs = xvgs
        self.nsim = nsim
        self.temp = temp
        self.save_dir = save_dir if save_dir else os.getcwd()
        self.subsample = subsample
        self.minpart = minpart
        self.maxpart = maxpart
        self.time_begin = time_begin
        self.time_end = time_end
        self.output_format = output_format
        self.overlap_threshold = overlap_threshold
        self.ess_threshold = ess_threshold
        self.n_windows = n_windows
        self.verbose = verbose
        self.tar_file = tar_file

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Data storage
        self.energies = None
        self.time_all = None
        self.nsamples = None
        self.eval_states = None
        self.u_kn = None
        self.n_k = None
        self.mb = None
        self.Deltaf_ij = None
        self.dDeltaf_ij = None
        self.overlap = None
        self.ess = None
        self.window_results = None
        self.window_centers = None

        if self.verbose:
            self._print_header()

    def _print_header(self):
        """Print analysis header."""
        print("="*60)
        print("MBAR Analysis for FEP Simulations")
        print("="*60)
        print(f"Temperature: {self.temp} K")
        print(f"Number of simulations: {self.nsim}")
        print(f"Save directory: {self.save_dir}")
        print("="*60)

    def load_data(self):
        """Load energy data from xvg files."""
        if self.verbose:
            print("\nLoading energy data...")
            if self.tar_file:
                print(f"Reading from tar file: {self.tar_file}")

        self.energies = {}
        self.time_all = {}
        self.nsamples = {}
        beta = 1. / (gasconstant * self.temp)

        # Determine all evaluation states
        eval_states = set()

        # Open tar file if specified
        tar_handle = None
        if self.tar_file:
            if not os.path.exists(self.tar_file):
                raise FileNotFoundError(f"Tar file not found: {self.tar_file}")
            tar_handle = tarfile.open(self.tar_file, 'r:gz')

        try:
            for isim in range(self.nsim):
                files = []
                if self.tar_file:
                    # Read from tar file
                    if self.minpart is not None:
                        for part in range(self.minpart, self.maxpart + 1):
                            tar_path = self.xvgs.replace("%sim", str(isim)).replace("%part", "%04d" % part)
                            try:
                                member = tar_handle.getmember(tar_path)
                                file_obj = tar_handle.extractfile(member)
                                if file_obj is not None:
                                    # Wrap in TextIOWrapper for text mode
                                    files.append(io.TextIOWrapper(file_obj, encoding='utf-8'))
                            except KeyError:
                                pass
                    else:
                        tar_path = self.xvgs.replace("%sim", str(isim))
                        try:
                            member = tar_handle.getmember(tar_path)
                            file_obj = tar_handle.extractfile(member)
                            if file_obj is not None:
                                # Wrap in TextIOWrapper for text mode
                                files.append(io.TextIOWrapper(file_obj, encoding='utf-8'))
                        except KeyError:
                            pass
                else:
                    # Read from regular files
                    if self.minpart is not None:
                        for part in range(self.minpart, self.maxpart + 1):
                            f = self.xvgs.replace("%sim", str(isim)).replace("%part", "%04d" % part)
                            if os.path.exists(f):
                                files.append(f)
                    else:
                        f = self.xvgs.replace("%sim", str(isim))
                        if os.path.exists(f):
                            files.append(f)

                if not files:
                    if self.verbose:
                        print(f"Warning: No files found for simulation {isim}", file=sys.stderr)
                    continue

                data = parse_deltae(files, self.subsample)

                # Close file objects if they were opened from tar
                if self.tar_file:
                    for f in files:
                        if hasattr(f, 'close'):
                            f.close()

                for (t, st, energy) in data:
                    eval_states.add(st)
                    se = SimEval(sim=isim, eval=st)
                    if se not in self.energies:
                        self.energies[se] = []
                        self.time_all[se] = []
                    if len(self.time_all[se]) > 0 and self.time_all[se][-1] == t:
                        # dup frame
                        continue
                    self.energies[se].append(energy)
                    self.time_all[se].append(t)

                self.nsamples[isim] = len(self.time_all.get(SimEval(sim=isim, eval=isim), []))

                if self.nsamples[isim] > 0 and self.verbose:
                    print(f"Finished loading simulation {isim}: {self.nsamples[isim]} samples from {len(files)} file(s)")
                    sys.stdout.flush()
        finally:
            if tar_handle:
                tar_handle.close()

        # Convert to numpy arrays and apply time filtering
        self.eval_states = sorted(eval_states)
        if self.verbose:
            print(f"Found evaluation states: {self.eval_states}")

        if self.time_begin is not None or self.time_end is not None:
            for k in list(self.energies.keys()):
                mask = np.ones(len(self.time_all[k]), dtype=bool)
                if self.time_begin is not None:
                    mask = mask & (self.time_all[k] >= self.time_begin)
                if self.time_end is not None:
                    mask = mask & (self.time_all[k] <= self.time_end)
                self.energies[k] = np.array(self.energies[k])[mask] * beta
                self.time_all[k] = np.array(self.time_all[k])[mask]
        else:
            for k in list(self.energies.keys()):
                self.energies[k] = np.array(self.energies[k]) * beta
                self.time_all[k] = np.array(self.time_all[k])

        if not self.energies:
            raise ValueError("No energy data loaded")

    def build_matrix(self):
        """Build u_kn matrix for MBAR analysis."""
        if self.verbose:
            print("\nBuilding u_kn matrix...")

        if self.energies is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Collect all samples: each sample is from a specific sim state
        # and has energies evaluated at multiple eval states
        all_samples = []

        # For each simulation state, collect samples
        for isim in range(self.nsim):
            base_se = SimEval(sim=isim, eval=isim)
            if base_se not in self.time_all:
                continue

            # Get time points for this simulation
            times = self.time_all[base_se]

            # For each time point, collect energies at available eval states
            for t_idx, t in enumerate(times):
                sample_energies = {}

                # Try to get energy at each eval state (may not all be available)
                for eval_state in self.eval_states:
                    se = SimEval(sim=isim, eval=eval_state)
                    if se in self.energies and se in self.time_all:
                        # Find matching time point
                        t_matches = np.abs(self.time_all[se] - t) < 0.1
                        if np.any(t_matches):
                            match_idx = np.where(t_matches)[0][0]
                            sample_energies[eval_state] = self.energies[se][match_idx]

                # Include sample if it has at least 2 states (minimum for MBAR)
                if len(sample_energies) >= 2:
                    all_samples.append({
                        'time': t,
                        'sim': isim,
                        'energies': sample_energies
                    })

        if not all_samples:
            raise ValueError("No valid samples found for MBAR analysis")

        if self.verbose:
            print(f"Total samples collected: {len(all_samples)}")

        # Determine which eval states are actually available across all samples
        available_states = set()
        for sample in all_samples:
            available_states.update(sample['energies'].keys())

        # Use only states that are available
        available_states = sorted([s for s in self.eval_states if s in available_states])
        if len(available_states) < 2:
            raise ValueError(f"Not enough states available for MBAR (found {len(available_states)} states)")

        if self.verbose:
            print(f"Available evaluation states: {available_states}")

        # Build u_kn matrix
        # K = number of evaluation states
        # N = number of samples
        K = len(available_states)
        N = len(all_samples)

        u_kn = np.full((K, N), np.nan)
        n_k = np.zeros(K, dtype=int)

        # Map evaluation state to index
        eval_to_idx = {available_states[i]: i for i in range(K)}

        # Fill u_kn matrix (NaN for missing energies)
        for n_idx, sample in enumerate(all_samples):
            for eval_state, energy in sample['energies'].items():
                if eval_state in eval_to_idx:
                    k_idx = eval_to_idx[eval_state]
                    u_kn[k_idx, n_idx] = energy

        # Count samples per state (samples that originated from each sim state)
        # Assume sim state i corresponds to eval state i if available
        for sample in all_samples:
            sim_state = sample['sim']
            # Find corresponding eval state index
            if sim_state in eval_to_idx:
                k_idx = eval_to_idx[sim_state]
                n_k[k_idx] += 1
            else:
                # Try to find closest available state
                for k_idx, eval_state in enumerate(available_states):
                    if abs(eval_state - sim_state) <= 1:  # Within 1 state
                        n_k[k_idx] += 1
                        break

        # For MBAR, we need samples that have energies at all states
        # However, in practice, not all samples may have all states
        # We'll use only samples that have at least some minimum number of states
        # and fill missing values with a large number (effectively excluding those states from that sample)

        # Count how many states each sample has
        n_states_per_sample = np.sum(~np.isnan(u_kn), axis=0)
        min_states_required = 2

        # Keep only samples with sufficient states
        valid_cols = n_states_per_sample >= min_states_required
        if np.sum(valid_cols) < N:
            if self.verbose:
                print(f"Warning: Removing {N - np.sum(valid_cols)} samples with insufficient state coverage")
            u_kn = u_kn[:, valid_cols]
            N = np.sum(valid_cols)

            # Recalculate n_k for filtered samples
            n_k = np.zeros(K, dtype=int)
            for i, sample in enumerate(all_samples):
                if valid_cols[i]:  # Only count samples that passed filtering
                    sim_state = sample['sim']
                    # Find corresponding eval state index
                    if sim_state in eval_to_idx:
                        k_idx = eval_to_idx[sim_state]
                        n_k[k_idx] += 1
                    else:
                        # Try to find closest available state
                        for k_idx, eval_state in enumerate(available_states):
                            if abs(eval_state - sim_state) <= 1:  # Within 1 state
                                n_k[k_idx] += 1
                                break

        # Check if any samples remain after filtering
        if N == 0:
            max_states_found = np.max(n_states_per_sample) if len(n_states_per_sample) > 0 else 0
            raise ValueError(
                f"No valid samples after filtering. "
                f"Required at least {min_states_required} states per sample, "
                f"but no samples met this requirement. "
                f"Maximum states per sample found: {max_states_found}"
            )

        # For MBAR, we need to handle NaN values
        # Fill NaN with large values (effectively infinite energy, zero probability)
        if N > 0:
            max_energy = np.nanmax(u_kn)
            min_energy = np.nanmin(u_kn)
            energy_range = max_energy - min_energy
            large_energy = max_energy + 10 * energy_range  # Large penalty for missing states

            u_kn_filled = np.where(np.isnan(u_kn), large_energy, u_kn)
        else:
            raise ValueError("No valid samples for MBAR analysis")

        if self.verbose:
            print(f"u_kn matrix shape: {u_kn_filled.shape} (K={K}, N={N})")
            print(f"Samples per state (n_k): {n_k}")
            print(f"NaN values filled with large energy: {large_energy:.2e}")

        self.u_kn = u_kn_filled
        self.n_k = n_k
        self.eval_states = available_states

    def compute_mbar(self):
        """Perform MBAR analysis."""
        if self.verbose:
            print("\n" + "="*60)
            print("Performing MBAR analysis...")
            print("="*60)

        if self.u_kn is None:
            raise ValueError("Matrix not built. Call build_matrix() first.")

        # Initialize MBAR
        try:
            self.mb = pymbar.MBAR(self.u_kn, self.n_k, verbose=self.verbose, relative_tolerance=1e-6)
        except Exception as e:
            print(f"Error initializing MBAR: {e}", file=sys.stderr)
            raise

        # Compute free energy differences
        if self.verbose:
            print("Computing free energy differences...")
        # pymbar 4.x API: compute_free_energy_differences()
        try:
            results = self.mb.compute_free_energy_differences(return_theta=False)
            self.Deltaf_ij = results['Delta_f']  # Free energy differences
            self.dDeltaf_ij = results['dDelta_f']  # Uncertainties
        except AttributeError:
            # Fallback for older pymbar versions
            results = self.mb.getFreeEnergyDifferences(compute_uncertainty=True)
            self.Deltaf_ij = results[0]  # Free energy differences
            self.dDeltaf_ij = results[1]  # Uncertainties

        # Compute overlap matrix
        if self.verbose:
            print("Computing overlap matrix...")
        # pymbar 4.x API: compute_overlap()
        try:
            self.overlap = self.mb.compute_overlap()
        except AttributeError:
            # Fallback for older pymbar versions
            self.overlap = self.mb.computeOverlap()

        # Compute effective sample sizes
        if self.verbose:
            print("Computing effective sample sizes...")
        # pymbar 4.x API: compute_effective_sample_number()
        try:
            self.ess = self.mb.compute_effective_sample_number()
        except AttributeError:
            # Fallback for older pymbar versions
            self.ess = self.mb.computeEffectiveSampleNumber()

    def plot_overlap_matrix(self):
        """Plot overlap matrix as heatmap."""
        if self.overlap is None:
            raise ValueError("MBAR analysis not performed. Call compute_mbar() first.")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract overlap matrix (first element of tuple)
        if isinstance(self.overlap, dict) and "matrix" in self.overlap:
            overlap_matrix = np.array(self.overlap["matrix"])
        elif isinstance(self.overlap, tuple):
            overlap_matrix = self.overlap[0]
        else:
            overlap_matrix = self.overlap

        im = ax.imshow(overlap_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(range(len(self.eval_states)))
        ax.set_yticks(range(len(self.eval_states)))
        ax.set_xticklabels([f"λ={s}" for s in self.eval_states], rotation=45, ha='right')
        ax.set_yticklabels([f"λ={s}" for s in self.eval_states])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Overlap', rotation=270, labelpad=20)

        # Add text annotations for low overlaps
        for i in range(len(self.eval_states)):
            for j in range(len(self.eval_states)):
                if overlap_matrix[i, j] < 0.01:
                    ax.text(j, i, f'{overlap_matrix[i, j]:.3f}',
                           ha='center', va='center', color='white', fontsize=6)

        ax.set_xlabel('State j')
        ax.set_ylabel('State i')
        ax.set_title('MBAR Overlap Matrix')

        plt.tight_layout()
        filename = os.path.join(self.save_dir, f'overlap_matrix.{self.output_format}')
        plt.savefig(filename, format=self.output_format, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Saved overlap matrix plot: {filename}")

    def plot_free_energy_curve(self):
        """Plot free energy curve."""
        if self.Deltaf_ij is None:
            raise ValueError("MBAR analysis not performed. Call compute_mbar() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to kcal/mol
        kcal_of_kJ = 1. / 4.184
        kBT = gasconstant * self.temp  # kJ/mol

        # Compute cumulative free energy from state 0
        n_states = len(self.eval_states)
        DeltaG = np.zeros(n_states)
        dDeltaG = np.zeros(n_states)

        for i in range(1, n_states):
            DeltaG[i] = self.Deltaf_ij[0, i] * kBT * kcal_of_kJ
            dDeltaG[i] = self.dDeltaf_ij[0, i] * kBT * kcal_of_kJ

        # Plot
        ax.errorbar(self.eval_states, DeltaG, yerr=dDeltaG, marker='o', linestyle='-',
                    capsize=3, capthick=1.5, linewidth=2, markersize=6)

        ax.set_xlabel('λ (State Index)')
        ax.set_ylabel('ΔG (kcal/mol)')
        ax.set_title('Free Energy Curve (relative to state 0)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(self.save_dir, f'free_energy_curve.{self.output_format}')
        plt.savefig(filename, format=self.output_format, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Saved free energy curve: {filename}")

    def plot_ess(self):
        """Plot effective sample sizes."""
        if self.ess is None:
            raise ValueError("MBAR analysis not performed. Call compute_mbar() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ESS values
        ess_values = np.array(self.ess)
        ess_ratio = ess_values / self.n_k

        # Plot ESS
        ax1.bar(range(len(self.eval_states)), ess_values, alpha=0.7, color='steelblue')
        ax1.axhline(y=np.mean(ess_values), color='r', linestyle='--', label=f'Mean: {np.mean(ess_values):.1f}')
        ax1.set_xlabel('State')
        ax1.set_ylabel('Effective Sample Size')
        ax1.set_title('Effective Sample Size per State')
        ax1.set_xticks(range(len(self.eval_states)))
        ax1.set_xticklabels([f"λ={s}" for s in self.eval_states], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot ESS ratio
        colors = ['red' if r < self.ess_threshold else 'steelblue' for r in ess_ratio]
        ax2.bar(range(len(self.eval_states)), ess_ratio, alpha=0.7, color=colors)
        ax2.axhline(y=self.ess_threshold, color='r', linestyle='--', label=f'Threshold: {self.ess_threshold}')
        ax2.set_xlabel('State')
        ax2.set_ylabel('ESS / N')
        ax2.set_title('ESS Ratio (ESS/N)')
        ax2.set_xticks(range(len(self.eval_states)))
        ax2.set_xticklabels([f"λ={s}" for s in self.eval_states], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = os.path.join(self.save_dir, f'effective_sample_size.{self.output_format}')
        plt.savefig(filename, format=self.output_format, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Saved ESS plot: {filename}")

    def plot_all(self):
        """Generate all plots."""
        self.plot_overlap_matrix()
        self.plot_free_energy_curve()
        self.plot_ess()

    def time_windowed_analysis(self):
        """Perform time-windowed MBAR analysis for convergence tracking."""
        if self.verbose:
            print("\n" + "="*60)
            print("Performing time-windowed convergence analysis...")
            print("="*60)

        if self.energies is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get time range
        all_times = []
        for isim in range(self.nsim):
            base_se = SimEval(sim=isim, eval=isim)
            if base_se in self.time_all:
                all_times.extend(self.time_all[base_se])

        if not all_times:
            if self.verbose:
                print("Warning: No time data found for convergence analysis")
            return

        tmin = min(all_times)
        tmax = max(all_times)
        if self.verbose:
            print(f"Time range: {tmin:.1f} - {tmax:.1f} ps")

        # Create time windows
        window_results = []
        window_centers = []

        for i in range(self.n_windows):
            t_start = tmin + (tmax - tmin) * i / self.n_windows
            t_end = tmin + (tmax - tmin) * (i + 1) / self.n_windows
            t_center = (t_start + t_end) / 2

            # Filter data for this window
            window_energies = {}
            window_time_all = {}

            for isim in range(self.nsim):
                for eval_state in self.eval_states:
                    se = SimEval(sim=isim, eval=eval_state)
                    if se in self.energies and se in self.time_all:
                        mask = (self.time_all[se] >= t_start) & (self.time_all[se] <= t_end)
                        if np.any(mask):
                            if se not in window_energies:
                                window_energies[se] = []
                                window_time_all[se] = []
                            window_energies[se].extend(self.energies[se][mask])
                            window_time_all[se].extend(self.time_all[se][mask])

            # Build u_kn for this window
            try:
                # Use a temporary instance to build matrix for this window
                temp_analyzer = MBARAnalysis(
                    self.xvgs, self.nsim, self.temp, self.save_dir,
                    self.subsample, self.minpart, self.maxpart,
                    self.time_begin, self.time_end, self.output_format,
                    self.overlap_threshold, self.ess_threshold, self.n_windows,
                    verbose=False
                )
                temp_analyzer.energies = window_energies
                temp_analyzer.time_all = window_time_all
                temp_analyzer.eval_states = self.eval_states
                temp_analyzer.build_matrix()

                if np.sum(temp_analyzer.n_k) > 0:
                    mb = pymbar.MBAR(temp_analyzer.u_kn, temp_analyzer.n_k, verbose=False, relative_tolerance=1e-6)
                    # pymbar 4.x API: compute_free_energy_differences()
                    try:
                        results = mb.compute_free_energy_differences(return_theta=False)
                        Deltaf_ij = results['Delta_f']
                        dDeltaf_ij = results['dDelta_f']
                    except AttributeError:
                        # Fallback for older pymbar versions
                        results = mb.getFreeEnergyDifferences(compute_uncertainty=True)
                        Deltaf_ij = results[0]
                        dDeltaf_ij = results[1]

                    # Total free energy difference
                    total_dG = Deltaf_ij[0, -1] * gasconstant * self.temp / 4.184  # kcal/mol
                    total_ddG = dDeltaf_ij[0, -1] * gasconstant * self.temp / 4.184  # kcal/mol

                    window_results.append((total_dG, total_ddG))
                    window_centers.append(t_center)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to analyze window {i}: {e}")
                continue

        if not window_results:
            if self.verbose:
                print("Warning: No valid windows for convergence analysis")
            return

        # Plot convergence
        fig, ax = plt.subplots(figsize=(12, 6))

        dG_vals = [r[0] for r in window_results]
        ddG_vals = [r[1] for r in window_results]

        ax.errorbar(window_centers, dG_vals, yerr=ddG_vals, marker='o', linestyle='-',
                    capsize=3, capthick=1.5, linewidth=2, markersize=6)

        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('ΔG (kcal/mol)')
        ax.set_title('Convergence of Free Energy Estimate')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(self.save_dir, f'convergence.{self.output_format}')
        plt.savefig(filename, format=self.output_format, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Saved convergence plot: {filename}")

        self.window_results = window_results
        self.window_centers = window_centers

    def print_summary(self):
        """Print and save summary statistics."""
        if self.Deltaf_ij is None:
            raise ValueError("MBAR analysis not performed. Call compute_mbar() first.")

        if self.verbose:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)

        kcal_of_kJ = 1. / 4.184
        kBT = gasconstant * self.temp

        # Free energy summary
        if self.verbose:
            print("\nFree Energy Differences (kcal/mol, relative to state 0):")
            print(f"{'State':<8} {'ΔG':<12} {'Uncertainty':<12}")
            print("-" * 32)
            for i, eval_state in enumerate(self.eval_states):
                dG = self.Deltaf_ij[0, i] * kBT * kcal_of_kJ
                ddG = self.dDeltaf_ij[0, i] * kBT * kcal_of_kJ
                print(f"λ={eval_state:<5} {dG:>10.4f} {ddG:>10.4f}")

        total_dG = self.Deltaf_ij[0, -1] * kBT * kcal_of_kJ
        total_ddG = self.dDeltaf_ij[0, -1] * kBT * kcal_of_kJ
        if self.verbose:
            print(f"\nTotal ΔG (state 0 -> {self.eval_states[-1]}): {total_dG:.4f} ± {total_ddG:.4f} kcal/mol")

        # Overlap summary
        if isinstance(self.overlap, dict) and "matrix" in self.overlap:
            overlap_matrix = np.array(self.overlap["matrix"])
        elif isinstance(self.overlap, tuple):
            overlap_matrix = self.overlap[0]
        else:
            overlap_matrix = self.overlap

        if self.verbose:
            print("\nOverlap Matrix (minimum values):")
            min_overlaps = []
            for i in range(len(self.eval_states) - 1):
                min_overlap = np.min(overlap_matrix[i, i+1])
                min_overlaps.append(min_overlap)
                print(f"  States {i}-{i+1}: {min_overlap:.4f}")

            min_overlap_overall = np.min(min_overlaps) if min_overlaps else 0.0
            print(f"\nMinimum overlap: {min_overlap_overall:.4f}")

            # ESS summary
            print("\nEffective Sample Sizes:")
            print(f"{'State':<8} {'ESS':<12} {'N':<12} {'ESS/N':<12}")
            print("-" * 44)
            for i, eval_state in enumerate(self.eval_states):
                ess_val = self.ess[i]
                n_val = self.n_k[i]
                ratio = ess_val / n_val if n_val > 0 else 0
                print(f"λ={eval_state:<5} {ess_val:>10.1f} {n_val:>10} {ratio:>10.4f}")
        else:
            min_overlap_overall = 0.0

        # Save to file
        summary_file = os.path.join(self.save_dir, 'mbar_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("MBAR Analysis Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Temperature: {self.temp} K\n")
            f.write(f"Number of states: {len(self.eval_states)}\n")
            f.write(f"States: {self.eval_states}\n\n")

            f.write("Free Energy Differences (kcal/mol, relative to state 0):\n")
            f.write(f"{'State':<8} {'ΔG':<12} {'Uncertainty':<12}\n")
            f.write("-" * 32 + "\n")
            for i, eval_state in enumerate(self.eval_states):
                dG = self.Deltaf_ij[0, i] * kBT * kcal_of_kJ
                ddG = self.dDeltaf_ij[0, i] * kBT * kcal_of_kJ
                f.write(f"λ={eval_state:<5} {dG:>10.4f} {ddG:>10.4f}\n")
            f.write(f"\nTotal ΔG: {total_dG:.4f} ± {total_ddG:.4f} kcal/mol\n\n")

            f.write(f"Minimum overlap: {min_overlap_overall:.4f}\n\n")

            f.write("Effective Sample Sizes:\n")
            f.write(f"{'State':<8} {'ESS':<12} {'N':<12} {'ESS/N':<12}\n")
            f.write("-" * 44 + "\n")
            for i, eval_state in enumerate(self.eval_states):
                ess_val = self.ess[i]
                n_val = self.n_k[i]
                ratio = ess_val / n_val if n_val > 0 else 0
                f.write(f"λ={eval_state:<5} {ess_val:>10.1f} {n_val:>10} {ratio:>10.4f}\n")

        if self.verbose:
            print(f"\nSummary saved to: {summary_file}")

    def save_data(self):
        """Save raw data to pickle file."""
        if self.Deltaf_ij is None:
            raise ValueError("MBAR analysis not performed. Call compute_mbar() first.")

        data_file = os.path.join(self.save_dir, 'mbar_data.pickle')
        with open(data_file, 'wb') as f:
            pickle.dump({
                'Deltaf_ij': self.Deltaf_ij,
                'dDeltaf_ij': self.dDeltaf_ij,
                'overlap': self.overlap,
                'ess': self.ess,
                'n_k': self.n_k,
                'eval_states': self.eval_states,
                'temp': self.temp
            }, f)

        if self.verbose:
            print(f"Raw data saved to: {data_file}")

    def run(self):
        """Run complete MBAR analysis pipeline."""
        self.load_data()
        self.build_matrix()
        self.compute_mbar()
        self.plot_all()
        self.time_windowed_analysis()
        self.print_summary()
        self.save_data()

        if self.verbose:
            print("\n" + "="*60)
            print("Analysis complete!")
            print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MBAR analysis for FEP simulations with convergence diagnostics'
    )
    parser.add_argument('--xvgs', metavar=".xvg", type=str, required=True,
                        help='xvg file path, "%%sim" will be replaced by simulation number')
    parser.add_argument('--nsim', metavar="N", type=int, required=True,
                        help='number of simulations (lambda windows)')
    parser.add_argument('--minpart', metavar="N", type=int, default=None,
                        help='part number begin')
    parser.add_argument('--maxpart', metavar="N", type=int, default=None,
                        help='part number end')
    parser.add_argument('--temp', help="Temperature (K)", type=float, default=300.0)
    parser.add_argument('--save-dir', help="save result to this directory",
                        type=str, default=os.getcwd())
    parser.add_argument('--subsample', help="subsample interval", type=int, default=1)
    parser.add_argument('--time-begin', help="beginning time (ps)", type=float, default=None)
    parser.add_argument('--time-end', help="end time (ps)", type=float, default=None)
    parser.add_argument('--output-format', help="output format for plots",
                        choices=['png', 'pdf', 'svg'], default='png')
    parser.add_argument('--overlap-threshold', help="minimum overlap threshold for warnings",
                        type=float, default=0.01)
    parser.add_argument('--ess-threshold', help="minimum ESS/N ratio for warnings",
                        type=float, default=0.1)
    parser.add_argument('--n-windows', help="number of time windows for convergence analysis",
                        type=int, default=10)
    parser.add_argument('--tar-file', help="Path to tar.gz file containing xvg files. "
                        "If specified, --xvgs should be the path pattern within the tar file",
                        type=str, default=None)

    opts = parser.parse_args()
    return opts


def main():
    """Main function for command-line interface."""
    opts = parse_args()

    analyzer = MBARAnalysis(
        xvgs=opts.xvgs,
        nsim=opts.nsim,
        temp=opts.temp,
        save_dir=opts.save_dir,
        subsample=opts.subsample,
        minpart=opts.minpart,
        maxpart=opts.maxpart,
        time_begin=opts.time_begin,
        time_end=opts.time_end,
        output_format=opts.output_format,
        overlap_threshold=opts.overlap_threshold,
        ess_threshold=opts.ess_threshold,
        n_windows=opts.n_windows,
        verbose=True,
        tar_file=opts.tar_file
    )

    analyzer.run()


if __name__ == "__main__":
    main()
