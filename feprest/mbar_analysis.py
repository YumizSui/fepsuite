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
import numpy
import os.path
import pickle
import collections
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import parse_deltae from bar_deltae
from bar_deltae import parse_deltae, SimEval, gasconstant, gasconstant_kcal

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def parse_args():
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

    opts = parser.parse_args()
    return opts


def load_energy_data(opts):
    """Load energy data from xvg files."""
    energies = {}
    time_all = {}
    nsamples = {}
    beta = 1. / (gasconstant * opts.temp)

    # Determine all evaluation states
    eval_states = set()

    for isim in range(opts.nsim):
        files = []
        if opts.minpart is not None:
            for part in range(opts.minpart, opts.maxpart + 1):
                f = opts.xvgs.replace("%sim", str(isim)).replace("%part", "%04d" % part)
                if os.path.exists(f):
                    files.append(f)
        else:
            f = opts.xvgs.replace("%sim", str(isim))
            if os.path.exists(f):
                files.append(f)

        if not files:
            print(f"Warning: No files found for simulation {isim}", file=sys.stderr)
            continue

        data = parse_deltae(files, opts.subsample)

        for (t, st, energy) in data:
            eval_states.add(st)
            se = SimEval(sim=isim, eval=st)
            if se not in energies:
                energies[se] = []
                time_all[se] = []
            if len(time_all[se]) > 0 and time_all[se][-1] == t:
                # dup frame
                continue
            energies[se].append(energy)
            time_all[se].append(t)

        nsamples[isim] = len(time_all.get(SimEval(sim=isim, eval=isim), []))

        if nsamples[isim] > 0:
            print(f"Finished loading simulation {isim}: {nsamples[isim]} samples from {len(files)} file(s)")
            sys.stdout.flush()

    # Convert to numpy arrays and apply time filtering
    eval_states = sorted(eval_states)
    print(f"Found evaluation states: {eval_states}")

    if opts.time_begin is not None or opts.time_end is not None:
        for k in list(energies.keys()):
            mask = numpy.ones(len(time_all[k]), dtype=bool)
            if opts.time_begin is not None:
                mask = mask & (time_all[k] >= opts.time_begin)
            if opts.time_end is not None:
                mask = mask & (time_all[k] <= opts.time_end)
            energies[k] = numpy.array(energies[k])[mask] * beta
            time_all[k] = numpy.array(time_all[k])[mask]
    else:
        for k in list(energies.keys()):
            energies[k] = numpy.array(energies[k]) * beta
            time_all[k] = numpy.array(time_all[k])

    return energies, time_all, nsamples, eval_states


def build_u_kn_matrix(energies, time_all, nsim, eval_states):
    """Build u_kn matrix for MBAR analysis.

    u_kn: reduced potential energy matrix
    K: number of states (evaluation states)
    N: total number of samples

    Each sample comes from a specific simulation state (sim),
    and we need the energy of that sample evaluated at all available states (eval).
    Note: Not all samples may have energies at all eval states.
    """
    # Collect all samples: each sample is from a specific sim state
    # and has energies evaluated at multiple eval states
    all_samples = []

    # For each simulation state, collect samples
    for isim in range(nsim):
        base_se = SimEval(sim=isim, eval=isim)
        if base_se not in time_all:
            continue

        # Get time points for this simulation
        times = time_all[base_se]

        # For each time point, collect energies at available eval states
        for t_idx, t in enumerate(times):
            sample_energies = {}

            # Try to get energy at each eval state (may not all be available)
            for eval_state in eval_states:
                se = SimEval(sim=isim, eval=eval_state)
                if se in energies and se in time_all:
                    # Find matching time point
                    t_matches = numpy.abs(time_all[se] - t) < 0.1
                    if numpy.any(t_matches):
                        match_idx = numpy.where(t_matches)[0][0]
                        sample_energies[eval_state] = energies[se][match_idx]

            # Include sample if it has at least 2 states (minimum for MBAR)
            if len(sample_energies) >= 2:
                all_samples.append({
                    'time': t,
                    'sim': isim,
                    'energies': sample_energies
                })

    if not all_samples:
        raise ValueError("No valid samples found for MBAR analysis")

    print(f"Total samples collected: {len(all_samples)}")

    # Determine which eval states are actually available across all samples
    available_states = set()
    for sample in all_samples:
        available_states.update(sample['energies'].keys())

    # Use only states that are available
    available_states = sorted([s for s in eval_states if s in available_states])
    if len(available_states) < 2:
        raise ValueError(f"Not enough states available for MBAR (found {len(available_states)} states)")

    print(f"Available evaluation states: {available_states}")

    # Build u_kn matrix
    # K = number of evaluation states
    # N = number of samples
    K = len(available_states)
    N = len(all_samples)

    u_kn = numpy.full((K, N), numpy.nan)
    n_k = numpy.zeros(K, dtype=int)

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
    n_states_per_sample = numpy.sum(~numpy.isnan(u_kn), axis=0)
    min_states_required = max(2, K // 2)  # At least 2 states, or half of available states

    # Keep only samples with sufficient states
    valid_cols = n_states_per_sample >= min_states_required
    if numpy.sum(valid_cols) < N:
        print(f"Warning: Removing {N - numpy.sum(valid_cols)} samples with insufficient state coverage")
        u_kn = u_kn[:, valid_cols]
        N = numpy.sum(valid_cols)

    # For MBAR, we need to handle NaN values
    # Option 1: Use only states that are present in all samples (very restrictive)
    # Option 2: Fill NaN with large values (effectively infinite energy, zero probability)
    # Option 3: Use pymbar's built-in handling if available

    # We'll use Option 2: fill NaN with a large value
    # This makes those state-sample combinations have effectively zero weight
    max_energy = numpy.nanmax(u_kn)
    min_energy = numpy.nanmin(u_kn)
    energy_range = max_energy - min_energy
    large_energy = max_energy + 10 * energy_range  # Large penalty for missing states

    u_kn_filled = numpy.where(numpy.isnan(u_kn), large_energy, u_kn)

    print(f"u_kn matrix shape: {u_kn_filled.shape} (K={K}, N={N})")
    print(f"Samples per state (n_k): {n_k}")
    print(f"NaN values filled with large energy: {large_energy:.2e}")

    return u_kn_filled, n_k, available_states


def compute_mbar_analysis(u_kn, n_k, verbose=True):
    """Perform MBAR analysis."""
    print("\n" + "="*60)
    print("Performing MBAR analysis...")
    print("="*60)

    # Initialize MBAR
    try:
        mb = pymbar.MBAR(u_kn, n_k, verbose=verbose, relative_tolerance=1e-6)
    except Exception as e:
        print(f"Error initializing MBAR: {e}", file=sys.stderr)
        raise

    # Compute free energy differences
    print("Computing free energy differences...")
    results = mb.getFreeEnergyDifferences(compute_uncertainty=True)
    Deltaf_ij = results[0]  # Free energy differences
    dDeltaf_ij = results[1]  # Uncertainties

    # Compute overlap matrix
    print("Computing overlap matrix...")
    overlap = mb.computeOverlap()

    # Compute effective sample sizes
    print("Computing effective sample sizes...")
    ess = mb.computeEffectiveSampleNumber()

    return mb, Deltaf_ij, dDeltaf_ij, overlap, ess


def plot_overlap_matrix(overlap, eval_states, save_dir, output_format='png'):
    """Plot overlap matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract overlap matrix (first element of tuple)
    if isinstance(overlap, tuple):
        overlap_matrix = overlap[0]
    else:
        overlap_matrix = overlap

    im = ax.imshow(overlap_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(eval_states)))
    ax.set_yticks(range(len(eval_states)))
    ax.set_xticklabels([f"λ={s}" for s in eval_states], rotation=45, ha='right')
    ax.set_yticklabels([f"λ={s}" for s in eval_states])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Overlap', rotation=270, labelpad=20)

    # Add text annotations for low overlaps
    for i in range(len(eval_states)):
        for j in range(len(eval_states)):
            if overlap_matrix[i, j] < 0.01:
                ax.text(j, i, f'{overlap_matrix[i, j]:.3f}',
                       ha='center', va='center', color='white', fontsize=6)

    ax.set_xlabel('State j')
    ax.set_ylabel('State i')
    ax.set_title('MBAR Overlap Matrix')

    plt.tight_layout()
    filename = os.path.join(save_dir, f'overlap_matrix.{output_format}')
    plt.savefig(filename, format=output_format, bbox_inches='tight')
    plt.close()
    print(f"Saved overlap matrix plot: {filename}")


def plot_free_energy_curve(Deltaf_ij, dDeltaf_ij, eval_states, save_dir, output_format='png', temp=300.0):
    """Plot free energy curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to kcal/mol
    kcal_of_kJ = 1. / 4.184
    kBT = gasconstant * temp  # kJ/mol

    # Compute cumulative free energy from state 0
    n_states = len(eval_states)
    DeltaG = numpy.zeros(n_states)
    dDeltaG = numpy.zeros(n_states)

    for i in range(1, n_states):
        DeltaG[i] = Deltaf_ij[0, i] * kBT * kcal_of_kJ
        dDeltaG[i] = dDeltaf_ij[0, i] * kBT * kcal_of_kJ

    # Plot
    ax.errorbar(eval_states, DeltaG, yerr=dDeltaG, marker='o', linestyle='-',
                capsize=3, capthick=1.5, linewidth=2, markersize=6)

    ax.set_xlabel('λ (State Index)')
    ax.set_ylabel('ΔG (kcal/mol)')
    ax.set_title('Free Energy Curve (relative to state 0)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(save_dir, f'free_energy_curve.{output_format}')
    plt.savefig(filename, format=output_format, bbox_inches='tight')
    plt.close()
    print(f"Saved free energy curve: {filename}")


def plot_ess(ess, n_k, eval_states, save_dir, output_format='png', threshold=0.1):
    """Plot effective sample sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ESS values
    ess_values = numpy.array(ess)
    ess_ratio = ess_values / n_k

    # Plot ESS
    ax1.bar(range(len(eval_states)), ess_values, alpha=0.7, color='steelblue')
    ax1.axhline(y=numpy.mean(ess_values), color='r', linestyle='--', label=f'Mean: {numpy.mean(ess_values):.1f}')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Effective Sample Size')
    ax1.set_title('Effective Sample Size per State')
    ax1.set_xticks(range(len(eval_states)))
    ax1.set_xticklabels([f"λ={s}" for s in eval_states], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot ESS ratio
    colors = ['red' if r < threshold else 'steelblue' for r in ess_ratio]
    ax2.bar(range(len(eval_states)), ess_ratio, alpha=0.7, color=colors)
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
    ax2.set_xlabel('State')
    ax2.set_ylabel('ESS / N')
    ax2.set_title('ESS Ratio (ESS/N)')
    ax2.set_xticks(range(len(eval_states)))
    ax2.set_xticklabels([f"λ={s}" for s in eval_states], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = os.path.join(save_dir, f'effective_sample_size.{output_format}')
    plt.savefig(filename, format=output_format, bbox_inches='tight')
    plt.close()
    print(f"Saved ESS plot: {filename}")


def time_windowed_analysis(energies, time_all, nsim, eval_states, n_windows, temp, save_dir, output_format='png'):
    """Perform time-windowed MBAR analysis for convergence tracking."""
    print("\n" + "="*60)
    print("Performing time-windowed convergence analysis...")
    print("="*60)

    # Get time range
    all_times = []
    for isim in range(nsim):
        base_se = SimEval(sim=isim, eval=isim)
        if base_se in time_all:
            all_times.extend(time_all[base_se])

    if not all_times:
        print("Warning: No time data found for convergence analysis")
        return

    tmin = min(all_times)
    tmax = max(all_times)
    print(f"Time range: {tmin:.1f} - {tmax:.1f} ps")

    # Create time windows
    window_results = []
    window_centers = []

    for i in range(n_windows):
        t_start = tmin + (tmax - tmin) * i / n_windows
        t_end = tmin + (tmax - tmin) * (i + 1) / n_windows
        t_center = (t_start + t_end) / 2

        # Filter data for this window
        window_energies = {}
        window_time_all = {}

        for isim in range(nsim):
            for eval_state in eval_states:
                se = SimEval(sim=isim, eval=eval_state)
                if se in energies and se in time_all:
                    mask = (time_all[se] >= t_start) & (time_all[se] <= t_end)
                    if numpy.any(mask):
                        if se not in window_energies:
                            window_energies[se] = []
                            window_time_all[se] = []
                        window_energies[se].extend(energies[se][mask])
                        window_time_all[se].extend(time_all[se][mask])

        # Build u_kn for this window
        try:
            u_kn, n_k, _ = build_u_kn_matrix(window_energies, window_time_all, nsim, eval_states)
            if numpy.sum(n_k) > 0:
                mb = pymbar.MBAR(u_kn, n_k, verbose=False, relative_tolerance=1e-6)
                results = mb.getFreeEnergyDifferences(compute_uncertainty=True)
                Deltaf_ij = results[0]
                dDeltaf_ij = results[1]

                # Total free energy difference
                total_dG = Deltaf_ij[0, -1] * gasconstant * temp / 4.184  # kcal/mol
                total_ddG = dDeltaf_ij[0, -1] * gasconstant * temp / 4.184  # kcal/mol

                window_results.append((total_dG, total_ddG))
                window_centers.append(t_center)
        except Exception as e:
            print(f"Warning: Failed to analyze window {i}: {e}")
            continue

    if not window_results:
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
    filename = os.path.join(save_dir, f'convergence.{output_format}')
    plt.savefig(filename, format=output_format, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {filename}")

    return window_results, window_centers


def print_summary(Deltaf_ij, dDeltaf_ij, overlap, ess, n_k, eval_states, temp, save_dir):
    """Print and save summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    kcal_of_kJ = 1. / 4.184
    kBT = gasconstant * temp

    # Free energy summary
    print("\nFree Energy Differences (kcal/mol, relative to state 0):")
    print(f"{'State':<8} {'ΔG':<12} {'Uncertainty':<12}")
    print("-" * 32)
    for i, eval_state in enumerate(eval_states):
        dG = Deltaf_ij[0, i] * kBT * kcal_of_kJ
        ddG = dDeltaf_ij[0, i] * kBT * kcal_of_kJ
        print(f"λ={eval_state:<5} {dG:>10.4f} {ddG:>10.4f}")

    total_dG = Deltaf_ij[0, -1] * kBT * kcal_of_kJ
    total_ddG = dDeltaf_ij[0, -1] * kBT * kcal_of_kJ
    print(f"\nTotal ΔG (state 0 -> {eval_states[-1]}): {total_dG:.4f} ± {total_ddG:.4f} kcal/mol")

    # Overlap summary
    if isinstance(overlap, tuple):
        overlap_matrix = overlap[0]
    else:
        overlap_matrix = overlap

    print("\nOverlap Matrix (minimum values):")
    min_overlaps = []
    for i in range(len(eval_states) - 1):
        min_overlap = numpy.min(overlap_matrix[i, i+1])
        min_overlaps.append(min_overlap)
        print(f"  States {i}-{i+1}: {min_overlap:.4f}")

    min_overlap_overall = numpy.min(min_overlaps)
    print(f"\nMinimum overlap: {min_overlap_overall:.4f}")

    # ESS summary
    print("\nEffective Sample Sizes:")
    print(f"{'State':<8} {'ESS':<12} {'N':<12} {'ESS/N':<12}")
    print("-" * 44)
    for i, eval_state in enumerate(eval_states):
        ess_val = ess[i]
        n_val = n_k[i]
        ratio = ess_val / n_val if n_val > 0 else 0
        print(f"λ={eval_state:<5} {ess_val:>10.1f} {n_val:>10} {ratio:>10.4f}")

    # Save to file
    summary_file = os.path.join(save_dir, 'mbar_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("MBAR Analysis Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Temperature: {temp} K\n")
        f.write(f"Number of states: {len(eval_states)}\n")
        f.write(f"States: {eval_states}\n\n")

        f.write("Free Energy Differences (kcal/mol, relative to state 0):\n")
        f.write(f"{'State':<8} {'ΔG':<12} {'Uncertainty':<12}\n")
        f.write("-" * 32 + "\n")
        for i, eval_state in enumerate(eval_states):
            dG = Deltaf_ij[0, i] * kBT * kcal_of_kJ
            ddG = dDeltaf_ij[0, i] * kBT * kcal_of_kJ
            f.write(f"λ={eval_state:<5} {dG:>10.4f} {ddG:>10.4f}\n")
        f.write(f"\nTotal ΔG: {total_dG:.4f} ± {total_ddG:.4f} kcal/mol\n\n")

        f.write(f"Minimum overlap: {min_overlap_overall:.4f}\n\n")

        f.write("Effective Sample Sizes:\n")
        f.write(f"{'State':<8} {'ESS':<12} {'N':<12} {'ESS/N':<12}\n")
        f.write("-" * 44 + "\n")
        for i, eval_state in enumerate(eval_states):
            ess_val = ess[i]
            n_val = n_k[i]
            ratio = ess_val / n_val if n_val > 0 else 0
            f.write(f"λ={eval_state:<5} {ess_val:>10.1f} {n_val:>10} {ratio:>10.4f}\n")

    print(f"\nSummary saved to: {summary_file}")


def main():
    opts = parse_args()

    # Create save directory
    os.makedirs(opts.save_dir, exist_ok=True)

    print("="*60)
    print("MBAR Analysis for FEP Simulations")
    print("="*60)
    print(f"Temperature: {opts.temp} K")
    print(f"Number of simulations: {opts.nsim}")
    print(f"Save directory: {opts.save_dir}")
    print("="*60)

    # Load data
    print("\nLoading energy data...")
    energies, time_all, nsamples, eval_states = load_energy_data(opts)

    if not energies:
        print("Error: No energy data loaded", file=sys.stderr)
        sys.exit(1)

    # Build u_kn matrix
    print("\nBuilding u_kn matrix...")
    u_kn, n_k, eval_states = build_u_kn_matrix(energies, time_all, opts.nsim, eval_states)

    # Perform MBAR analysis
    mb, Deltaf_ij, dDeltaf_ij, overlap, ess = compute_mbar_analysis(u_kn, n_k)

    # Generate plots
    print("\nGenerating plots...")
    plot_overlap_matrix(overlap, eval_states, opts.save_dir, opts.output_format)
    plot_free_energy_curve(Deltaf_ij, dDeltaf_ij, eval_states, opts.save_dir,
                          opts.output_format, opts.temp)
    plot_ess(ess, n_k, eval_states, opts.save_dir, opts.output_format, opts.ess_threshold)

    # Time-windowed convergence analysis
    time_windowed_analysis(energies, time_all, opts.nsim, eval_states, opts.n_windows,
                          opts.temp, opts.save_dir, opts.output_format)

    # Print and save summary
    print_summary(Deltaf_ij, dDeltaf_ij, overlap, ess, n_k, eval_states, opts.temp, opts.save_dir)

    # Save raw data
    data_file = os.path.join(opts.save_dir, 'mbar_data.pickle')
    with open(data_file, 'wb') as f:
        pickle.dump({
            'Deltaf_ij': Deltaf_ij,
            'dDeltaf_ij': dDeltaf_ij,
            'overlap': overlap,
            'ess': ess,
            'n_k': n_k,
            'eval_states': eval_states,
            'temp': opts.temp
        }, f)
    print(f"Raw data saved to: {data_file}")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()

