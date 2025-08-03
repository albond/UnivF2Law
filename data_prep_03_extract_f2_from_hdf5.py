#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Extract f2 frequencies from CoRe HDF5 files using Welch's method.
This script reads gravitational wave data from HDF5 files and extracts
the dominant post-merger oscillation frequency (f2).

Version 3.0 - Welch method:
- Uses Welch's method instead of FFT for more robust spectral estimation
- Automatically averages over segments to reduce noise
- Better suppression of spurious peaks
- Improved frequency resolution estimation
- Maintains compatibility with existing data format
"""

import numpy as np
import pandas as pd
import h5py
import os
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def frequency_domain_integration(time, data, integration_order=1):
    """
    Perform integration in frequency domain to reduce low-frequency noise.
    
    Parameters:
    -----------
    time : array
        Time array
    data : array
        Data to integrate (complex)
    integration_order : int
        Number of integrations (1 or 2)
    
    Returns:
    --------
    integrated : array
        Integrated data
    """
    dt = np.mean(np.diff(time))
    n = len(data)
    
    # FFT of the data
    data_fft = fft(data)
    freqs = fftfreq(n, dt)
    
    # Avoid division by zero at DC component
    # Apply high-pass filter to remove low-frequency drift
    f_min = 10.0  # Hz, minimum frequency to consider
    omega = 2 * np.pi * freqs
    
    # Create integration operator in frequency domain
    # Integration is division by iω in frequency domain
    # Double integration is division by (iω)²
    with np.errstate(divide='ignore', invalid='ignore'):
        if integration_order == 1:
            # Single integration: divide by iω
            integrator = np.where(np.abs(freqs) > f_min, 1.0 / (1j * omega), 0)
        elif integration_order == 2:
            # Double integration: divide by (iω)²
            integrator = np.where(np.abs(freqs) > f_min, -1.0 / (omega**2), 0)
        else:
            raise ValueError("integration_order must be 1 or 2")
    
    # Apply integration in frequency domain
    integrated_fft = data_fft * integrator
    
    # Transform back to time domain
    integrated = ifft(integrated_fft)
    
    # Return real part if input was real, otherwise return complex
    if np.isrealobj(data):
        return np.real(integrated)
    else:
        return integrated

def find_merger_time(time, strain):
    """Find the merger time based on the peak amplitude of the strain."""
    # Compute the amplitude envelope
    amplitude = np.abs(strain)
    
    # Find the peak (merger)
    merger_idx = np.argmax(amplitude)
    merger_time = time[merger_idx]
    
    return merger_time, merger_idx

def extract_f2_frequency(time, strain, merger_time=None, plot=False, sim_name=""):
    """
    Extract f2 frequency from post-merger gravitational wave signal.
    
    Parameters:
    -----------
    time : array
        Time array in geometric units (M_sun)
    strain : array
        Gravitational wave strain (complex)
    merger_time : float
        Time of merger (if None, will be detected automatically)
    plot : bool
        Whether to create diagnostic plots
    sim_name : str
        Simulation name for plot titles
    
    Returns:
    --------
    tuple : (f2_freq, f2_err, quality_flag, peak_to_median, n_sigma)
        f2_freq : float
            f2 frequency in Hz
        f2_err : float
            Estimated uncertainty in Hz
        quality_flag : int
            0: high quality, 1: weak peak, 2: multi-peak, 3: both issues
        peak_to_median : float
            Ratio of peak power to median power
        n_sigma : float
            Number of standard deviations above background
    """

    code_family = "BAM" if "BAM_" in sim_name else ("THC" if "THC_" in sim_name else "")
    if merger_time is None:
        merger_time, merger_idx = find_merger_time(time, strain)
    else:
        merger_idx = np.argmin(np.abs(time - merger_time))
    
    # Select post-merger data (typically 5-20 ms after merger)
    # Convert time from geometric units to ms (1 M_sun ~ 4.925e-6 s)
    time_ms = time * 4.925e-3  # Convert to ms
    merger_time_ms = merger_time * 4.925e-3

    # ------------------------------------------------------------
    # 2) Adaptive post‑merger window with code‑specific offset
    #    BAM simulations are typically shorter and need a smaller
    #    offset; THC are longer and more stable, so keep 2 ms.
    # ------------------------------------------------------------
    offset_ms = 1.0 if code_family == "BAM" else 2.0

    # Determine available remnant signal duration
    time_end_ms = time_ms[-1]
    available_duration = time_end_ms - merger_time_ms

    post_merger_start = merger_time_ms + offset_ms
    post_merger_end = min(merger_time_ms + 20.0,  # hard cap 20 ms
                          merger_time_ms + 0.8 * available_duration)
    
    # Ensure we have at least 5ms of data
    if post_merger_end - post_merger_start < 5.0:
        print(f"Warning: Short post-merger duration ({post_merger_end - post_merger_start:.1f} ms) for {sim_name}")
    
    # Find indices for post-merger window
    post_merger_mask = (time_ms >= post_merger_start) & (time_ms <= post_merger_end)
    
    if np.sum(post_merger_mask) < 100:
        print(f"Warning: Not enough post-merger data for {sim_name}")
        return None, None
    
    # Extract post-merger data
    time_pm = time[post_merger_mask]
    strain_pm = strain[post_merger_mask]
    
    # Remove any linear trend
    strain_pm_real = np.real(strain_pm)
    strain_pm_imag = np.imag(strain_pm)
    
    # Detrend
    strain_pm_real = signal.detrend(strain_pm_real)
    strain_pm_imag = signal.detrend(strain_pm_imag)
    
    # Apply window to reduce edge effects
    # Use Tukey window for better spectral properties
    try:
        window = signal.windows.tukey(len(strain_pm_real), alpha=0.2)
    except AttributeError:
        # Fallback to Hanning if Tukey not available
        window = np.hanning(len(strain_pm_real))
    
    strain_pm_real *= window
    strain_pm_imag *= window
    
    # Compute using Welch's method
    dt = np.mean(np.diff(time_pm))  # in geometric units
    dt_s = dt * 4.925e-6  # Convert to seconds
    fs = 1.0 / dt_s  # Sampling frequency in Hz

    # ------------------------------------------------------------
    # 3) Welch parameters — tune nperseg by family.
    #    THC runs are typically longer → use more points per segment
    # ------------------------------------------------------------
    if code_family == "THC":
        nperseg = max(512, len(strain_pm_real) // 6)
    else:  # BAM or unknown
        nperseg = max(256, len(strain_pm_real) // 4)

    # Ensure nperseg is not larger than the signal length
    if nperseg > len(strain_pm_real):
        nperseg = len(strain_pm_real) // 2
    
    # Use 50% overlap
    noverlap = nperseg // 2
    
    # Apply Welch's method to both real and imaginary parts
    frequencies_real, psd_real = signal.welch(strain_pm_real, fs=fs, window='hann', 
                                            nperseg=nperseg, noverlap=noverlap,
                                            detrend='constant')
    
    frequencies_imag, psd_imag = signal.welch(strain_pm_imag, fs=fs, window='hann',
                                            nperseg=nperseg, noverlap=noverlap,
                                            detrend='constant')
    
    # Combine power spectra (for complex signal, total power is sum of components)
    frequencies_pos = frequencies_real
    power_spectrum = psd_real + psd_imag
    
    # No need for additional smoothing - Welch's method already provides smoothed spectrum
    power_spectrum_smooth = power_spectrum
    
    # Adaptive f2 frequency range based on total mass
    # Lower masses have higher f2 frequencies
    # Default range: 1000-5000 Hz, adaptive based on M_total if available
    f2_min = 1000  # Hz
    f2_max = 5000  # Hz
    
    # TODO: Could make this adaptive based on M_total and Lambda_tilde
    # For now, use wider search range as recommended
    f2_mask = (frequencies_pos >= f2_min) & (frequencies_pos <= f2_max)
    f2_frequencies = frequencies_pos[f2_mask]
    f2_power = power_spectrum_smooth[f2_mask] if 'power_spectrum_smooth' in locals() else power_spectrum[f2_mask]
    
    if len(f2_frequencies) == 0:
        print(f"Warning: No frequencies in f2 range for {sim_name}")
        return None, None
    
    # Find the global maximum in the f2 range
    peak_idx = np.argmax(f2_power)
    f2_freq = f2_frequencies[peak_idx]
    peak_power = f2_power[peak_idx]
    
    # Enhanced quality check with multiple criteria
    # 1. Peak should be at least 3x the median power
    median_power = np.median(f2_power)
    peak_to_median = peak_power / median_power if median_power > 0 else 0
    
    # 2. Peak should exceed background by n standard deviations
    std_power = np.std(f2_power)
    mean_power = np.mean(f2_power)
    n_sigma = (peak_power - mean_power) / std_power if std_power > 0 else 0
    
    # 3. Multi-peak analysis (check for ambiguity)
    # Find all peaks above 50% of the main peak
    secondary_peaks = []
    for i in range(1, len(f2_power) - 1):
        if (f2_power[i] > f2_power[i-1] and f2_power[i] > f2_power[i+1] and 
            f2_power[i] > 0.5 * peak_power and i != peak_idx):
            secondary_peaks.append((f2_frequencies[i], f2_power[i]))
    
    # Determine quality flag
    quality_flag = 0
    weak_peak = (peak_to_median < 3.0) or (n_sigma < 3.0)  # Restored strict criteria
    multi_peak = len(secondary_peaks) > 0
    
    if weak_peak and multi_peak:
        quality_flag = 3
    elif multi_peak:
        quality_flag = 2
    elif weak_peak:
        quality_flag = 1
    
    # Adjust uncertainty based on quality
    if quality_flag == 0:
        f2_err_factor = 1.0
    else:
        # Increase uncertainty for low-quality data
        f2_err_factor = max(2.0, 1000.0 / f2_freq) if f2_freq > 0 else 10.0
    
    # Estimate uncertainty based on frequency resolution and peak width
    # For Welch's method, frequency resolution is fs / nperseg
    freq_resolution = fs / nperseg
    
    # Include noise level in uncertainty estimate
    noise_contribution = freq_resolution * (median_power / peak_power) if peak_power > 0 else 0
    
    # Find FWHM of the peak for better uncertainty estimate
    half_max = peak_power / 2
    # Find indices where power > half_max around the peak
    peak_region = np.where(f2_power > half_max)[0]
    if len(peak_region) > 1:
        # Find contiguous region around peak
        peak_in_region = np.where(peak_region == peak_idx)[0]
        if len(peak_in_region) > 0:
            # Find connected component containing the peak
            peak_pos = peak_in_region[0]
            start = peak_pos
            end = peak_pos
            # Expand left
            while start > 0 and peak_region[start] - peak_region[start-1] == 1:
                start -= 1
            # Expand right  
            while end < len(peak_region)-1 and peak_region[end+1] - peak_region[end] == 1:
                end += 1
            
            fwhm_indices = peak_region[end] - peak_region[start] + 1
            fwhm_hz = fwhm_indices * (f2_frequencies[1] - f2_frequencies[0]) if len(f2_frequencies) > 1 else freq_resolution
            # Include all sources of uncertainty
            resolution_err = freq_resolution
            fwhm_err = fwhm_hz / 2.355  # Convert FWHM to sigma
            noise_err = noise_contribution
            f2_err = np.sqrt(resolution_err**2 + fwhm_err**2 + noise_err**2) * f2_err_factor
        else:
            f2_err = np.sqrt(freq_resolution**2 + noise_contribution**2) * 2 * f2_err_factor
    else:
        f2_err = np.sqrt(freq_resolution**2 + noise_contribution**2) * 2 * f2_err_factor
    
    # Create diagnostic plot if requested
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Full waveform
        ax = axes[0, 0]
        ax.plot(time_ms, np.abs(strain), 'b-', alpha=0.7)
        ax.axvline(merger_time_ms, color='r', linestyle='--', label='Merger')
        ax.axvspan(post_merger_start, post_merger_end, alpha=0.3, color='green', label='Analysis window')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|h|')
        ax.set_title(f'{sim_name}: Full Waveform')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Post-merger waveform
        ax = axes[0, 1]
        time_pm_ms = time_pm * 4.925e-3
        ax.plot(time_pm_ms - merger_time_ms, strain_pm_real, 'b-', alpha=0.7, label='Real')
        ax.plot(time_pm_ms - merger_time_ms, strain_pm_imag, 'r-', alpha=0.7, label='Imag')
        ax.set_xlabel('Time after merger (ms)')
        ax.set_ylabel('h')
        ax.set_title('Post-merger Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Full power spectrum
        ax = axes[1, 0]
        ax.semilogy(frequencies_pos[frequencies_pos < 5000], power_spectrum[frequencies_pos < 5000])
        ax.axvline(f2_freq, color='r', linestyle='--', label=f'f2 = {f2_freq:.0f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('Power Spectrum (Welch)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed f2 region
        ax = axes[1, 1]
        ax.plot(f2_frequencies, f2_power)
        ax.axvline(f2_freq, color='r', linestyle='--', label=f'f2 = {f2_freq:.0f} ± {f2_err:.0f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('f2 Region (Welch)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figs/f2_extraction_{sim_name.replace(":", "_")}.png', dpi=150)
        plt.close()
    
    return f2_freq, f2_err, quality_flag, peak_to_median, n_sigma

def process_hdf5_file(hdf5_path, sim_info, plot_first_few=False):
    """
    Process a single HDF5 file to extract f2 frequency.
    
    Parameters:
    -----------
    hdf5_path : str
        Path to the HDF5 file
    sim_info : dict
        Simulation information from ./data/best_core_simulations.csv
    plot_first_few : bool
        Whether to create plots for diagnostic
    
    Returns:
    --------
    result : dict or None
        Dictionary with extracted f2 data or None if extraction failed
    """
    
    sim_name = sim_info['simulation']
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Look for gravitational wave data groups or datasets
            gw_data = None
            extraction_radius = None
            
            # Check for group structure (BAM format)
            if 'rh_22' in f:
                # Get the group
                rh_group = f['rh_22']
                # Find the largest extraction radius (usually best)
                radii = []
                for key in rh_group.keys():
                    if 'Rh_l2_m2_r' in key:
                        # Extract radius from filename like 'Rh_l2_m2_r00900.txt'
                        try:
                            r_str = key.split('_r')[-1].split('.')[0]
                            r = int(r_str) / 100.0  # Convert to M_sun units
                            radii.append((r, key))
                        except:
                            continue
                
                if radii:
                    # Use the largest radius
                    radii.sort()
                    extraction_radius, data_key = radii[-1]
                    gw_data = rh_group[data_key][()]
                    data_type = 'rh'
                else:
                    print(f"Warning: No Rh data found in rh_22 group for {sim_name}")
                    return None
                    
            elif 'rpsi4_22' in f:
                # Similar for psi4 data
                psi4_group = f['rpsi4_22']
                radii = []
                for key in psi4_group.keys():
                    if 'Rpsi4_l2_m2_r' in key:
                        try:
                            r_str = key.split('_r')[-1].split('.')[0]
                            r = int(r_str) / 100.0
                            radii.append((r, key))
                        except:
                            continue
                
                if radii:
                    radii.sort()
                    extraction_radius, data_key = radii[-1]
                    gw_data = psi4_group[data_key][()]
                    data_type = 'psi4'
                else:
                    print(f"Warning: No psi4 data found in rpsi4_22 group for {sim_name}")
                    return None
            else:
                # Try direct dataset access (THC format might be different)
                possible_datasets = ['rhOverM_22', 'psi4_22', 'h_22', 'strain_22']
                for ds_name in possible_datasets:
                    if ds_name in f:
                        gw_data = f[ds_name][()]
                        # Fixed: Force rh type for h_22 and strain_22 to avoid double integration
                        if 'h_22' in ds_name or 'strain_22' in ds_name or 'rh' in ds_name:
                            data_type = 'rh'
                        else:
                            data_type = 'psi4'
                        extraction_radius = 100.0  # Default assumption
                        break
                
                if gw_data is None:
                    print(f"Warning: No GW data found in {sim_name}")
                    return None
            
            # Now process the gw_data we found
            data = gw_data
            
            # Parse the data format based on shape
            # BAM format typically has columns: [time, h_real, h_imag, ...]
            # with 9 columns for rh data or 7 for psi4 data
            if len(data.shape) == 2:
                if data.shape[1] >= 3:
                    # First column is time
                    time = data[:, 0]
                    # For rh data: columns are typically [time, h+, hx, ...]
                    # For psi4 data: similar structure
                    if data_type == 'rh':
                        # Use columns 1 and 2 as real and imaginary parts
                        strain = data[:, 1] + 1j * data[:, 2]
                    else:  # psi4
                        strain = data[:, 1] + 1j * data[:, 2]
                else:
                    print(f"Warning: Unexpected data shape in {sim_name}: {data.shape}")
                    return None
            else:
                print(f"Warning: Expected 2D data array in {sim_name}, got shape: {data.shape}")
                return None
            
            # For psi4 data, we need to integrate twice to get strain
            if data_type == 'psi4':
                # Improved frequency-domain integration
                strain = frequency_domain_integration(time, strain, integration_order=2)
            
            # Extract f2
            plot = plot_first_few and (sim_info.get('quality_score', 0) > 25)
            extraction_result = extract_f2_frequency(time, strain, plot=plot, sim_name=sim_name)
            
            if extraction_result is None or extraction_result[0] is None:
                return None
            
            f2_freq, f2_err, quality_flag, peak_to_median, n_sigma = extraction_result
            
            # Prepare result
            result = {
                'eos': sim_info.get('eos', sim_info.get('eos1', 'Unknown')),
                'm1': sim_info.get('mass1', 0.0),
                'm2': sim_info.get('mass2', 0.0),
                'f2_Hz': int(round(f2_freq)),
                'f2_err_Hz': int(round(f2_err)),
                'note': f'CoRe {sim_name}',
                'source': 'CoRe_Welch_v3',
                'M_total': sim_info.get('mass1', 0.0) + sim_info.get('mass2', 0.0),
                'q': min(sim_info.get('mass1', 1.0), sim_info.get('mass2', 1.0)) / 
                     max(sim_info.get('mass1', 1.0), sim_info.get('mass2', 1.0)),
                'Lambda1': sim_info.get('Lambda1', None),
                'Lambda2': sim_info.get('Lambda2', None),
                'Lambda_tilde': sim_info.get('Lambda', None),
                'simulation': sim_name,
                'quality_flag': quality_flag,
                'peak_to_median': round(peak_to_median, 2),
                'n_sigma': round(n_sigma, 2),
                'pass_strict_filter': quality_flag == 0
            }
            
            return result
            
    except Exception as e:
        print(f"Error processing {sim_name}: {e}")
        return None

def read_metadata_file(metadata_path):
    """Read and parse metadata_main.txt file."""
    metadata = {}
    try:
        with open(metadata_path, 'r') as f:
            content = f.read()
            
            # Parse key fields using regex
            import re
            
            # EoS
            eos_match = re.search(r'id_eos\s*=\s*(\w+)', content)
            if eos_match:
                metadata['eos'] = eos_match.group(1)
            
            # Masses
            mass1_match = re.search(r'id_mass_starA\s*=\s*([\d.]+)', content)
            mass2_match = re.search(r'id_mass_starB\s*=\s*([\d.]+)', content)
            if mass1_match and mass2_match:
                metadata['m1'] = float(mass1_match.group(1))
                metadata['m2'] = float(mass2_match.group(1))
            
            # Lambda values
            lambda1_match = re.search(r'id_Lambdaell_starA\s*=\s*([\d.]+)', content)
            lambda2_match = re.search(r'id_Lambdaell_starB\s*=\s*([\d.]+)', content)
            if lambda1_match:
                metadata['Lambda1'] = float(lambda1_match.group(1))
            if lambda2_match:
                metadata['Lambda2'] = float(lambda2_match.group(1))
                
            # Combined Lambda
            lambda_match = re.search(r'id_Lambda\s*=\s*([\d.]+)', content)
            if lambda_match:
                metadata['Lambda'] = float(lambda_match.group(1))
                
    except Exception as e:
        pass  # Silently continue with defaults
    
    return metadata

def main():
    """Main function to extract f2 from all downloaded HDF5 files."""
    
    print("Extracting f2 frequencies from CoRe HDF5 files...")
    print("=" * 60)
    
    # Create output directory for plots
    os.makedirs('figs', exist_ok=True)
    
    # Find all downloaded HDF5 files
    hdf5_files = []
    core_dir = 'CoRe'
    
    if not os.path.exists(core_dir):
        print(f"Error: {core_dir} directory not found!")
        return
        
    # Scan for all data.h5 files in all subdirectories
    for sim_dir in os.listdir(core_dir):
        sim_path = os.path.join(core_dir, sim_dir)
        if os.path.isdir(sim_path):
            # Check if data.h5 exists directly in this directory
            hdf5_path = os.path.join(sim_path, 'data.h5')
            if os.path.exists(hdf5_path):
                # Get metadata
                metadata_path = os.path.join(sim_path, 'metadata_main.txt')
                metadata = read_metadata_file(metadata_path)
                
                # Create sim_info dict
                sim_info = {
                    'simulation': sim_dir.replace('_', ':'),
                    'eos': metadata.get('eos', 'unknown'),
                    'eos1': metadata.get('eos', 'unknown'),
                    'mass1': metadata.get('m1', 1.35),
                    'mass2': metadata.get('m2', 1.35),
                    'Lambda1': metadata.get('Lambda1'),
                    'Lambda2': metadata.get('Lambda2'),
                    'Lambda': metadata.get('Lambda'),
                    'quality_score': 20  # Default score
                }
                
                hdf5_files.append((hdf5_path, sim_info))
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process files
    results = []
    successful = 0
    failed = []
    
    # Process first few with plots for diagnostics
    plot_count = 0
    max_plots = 5
    
    print("\nProcessing HDF5 files...")
    print("-" * 60)
    
    # Use sequential processing for better error tracking
    for i, (hdf5_path, sim_info) in enumerate(hdf5_files):
        sim_name = sim_info['simulation']
        print(f"[{i+1}/{len(hdf5_files)}] Processing {sim_name}...", end='', flush=True)
        
        plot = (plot_count < max_plots) and (sim_info.get('quality_score', 0) > 25)
        result = process_hdf5_file(hdf5_path, sim_info, plot_first_few=plot)
        
        if result is not None:
            results.append(result)
            successful += 1
            print(f" ✓ f2 = {result['f2_Hz']} Hz")
            if plot:
                plot_count += 1
        else:
            failed.append(sim_name)
            print(" ✗ Failed")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Extraction Summary:")
    print(f"  Successful: {successful}/{len(hdf5_files)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed simulations:")
        for sim in failed[:10]:  # Show first 10
            print(f"  - {sim}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        
        # Sort by EoS and total mass
        df = df.sort_values(['eos', 'M_total'])
        
        # Save to CSV with quality information
        output_columns = ['eos', 'm1', 'm2', 'f2_Hz', 'f2_err_Hz', 'note', 'source', 'M_total', 'q', 
                         'quality_flag', 'peak_to_median', 'n_sigma', 'pass_strict_filter']
        df[output_columns].to_csv('./data/integrated_f2_database.csv', index=False)
        
        print(f"\nSaved {len(df)} f2 measurements to ./data/integrated_f2_database.csv")
        print(f"\nNote: This file contains f2 data extracted from CoRe HDF5 files.")
        print("To use literature-based data instead, run: python3 integrated_real_f2_data.py")
        
        # Print statistics
        print("\nExtracted f2 Statistics:")
        print(f"  f2 range: {df['f2_Hz'].min()} - {df['f2_Hz'].max()} Hz")
        print(f"  Mean f2: {df['f2_Hz'].mean():.0f} ± {df['f2_Hz'].std():.0f} Hz")
        print(f"  Unique EoS: {df['eos'].nunique()}")
        
        
        # Create a summary plot
        plt.figure(figsize=(10, 6))
        for eos in df['eos'].unique()[:10]:  # Plot first 10 EoS
            eos_data = df[df['eos'] == eos]
            plt.scatter(eos_data['M_total'], eos_data['f2_Hz'], label=eos, alpha=0.7)
        
        plt.xlabel('Total Mass (M☉)')
        plt.ylabel('f2 Frequency (Hz)')
        plt.title('Extracted f2 Frequencies from CoRe HDF5 Data')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figs/f2_extracted_summary.png', dpi=150)
        plt.close()
        
        print("\nCreated summary plot: figs/f2_extracted_summary.png")

if __name__ == "__main__":
    main()