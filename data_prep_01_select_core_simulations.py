#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Select best CoRe simulations based on metadata_main.txt files.
This script filters BAM and THC simulations to find those with:
1. Binary neutron star mergers (BNS)
2. Sufficient resolution
3. Long enough post-merger evolution
4. Available gravitational wave data
"""

import requests
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os


def parse_metadata(text):
    """Parse metadata_main.txt content to extract key parameters."""
    metadata = {}
    
    # Extract binary type
    match = re.search(r'binary_type\s*=\s*(\w+)', text)
    if match:
        metadata['binary_type'] = match.group(1)
    
    # Extract masses
    mass1_match = re.search(r'id_mass_starA\s*=\s*([\d.e+-]+)', text)
    mass2_match = re.search(r'id_mass_starB\s*=\s*([\d.e+-]+)', text)
    if mass1_match and mass2_match:
        metadata['mass1'] = float(mass1_match.group(1))
        metadata['mass2'] = float(mass2_match.group(1))
    
    # Extract EoS
    eos_match = re.search(r'id_eos\s*=\s*(\w+)', text)
    if eos_match:
        metadata['eos'] = eos_match.group(1)
        metadata['eos1'] = eos_match.group(1)
        metadata['eos2'] = eos_match.group(1)
    
    # Extract Lambda (tidal deformability)
    lambda_match = re.search(r'id_Lambda\s*=\s*([\d.e+-]+)', text)
    if lambda_match:
        metadata['Lambda'] = float(lambda_match.group(1))
    
    # Extract individual Lambdas
    lambda1_match = re.search(r'id_Lambdaell_starA\s*=\s*([\d.e+-]+)', text)
    lambda2_match = re.search(r'id_Lambdaell_starB\s*=\s*([\d.e+-]+)', text)
    if lambda1_match:
        metadata['Lambda1'] = float(lambda1_match.group(1))
    if lambda2_match:
        metadata['Lambda2'] = float(lambda2_match.group(1))
    
    # Extract eccentricity
    ecc_match = re.search(r'id_eccentricity\s*=\s*([\d.e+-]+)', text)
    if ecc_match:
        metadata['eccentricity'] = float(ecc_match.group(1))
    
    # Extract database key
    key_match = re.search(r'database_key\s*=\s*(\w+:\d+)', text)
    if key_match:
        metadata['database_key'] = key_match.group(1)
    
    # Check available runs
    runs_match = re.search(r'available_runs\s*=\s*([^\n]+)', text)
    if runs_match:
        runs_str = runs_match.group(1).strip()
        # Remove comments or extra text
        runs_str = runs_str.split('#')[0].strip()
        metadata['available_runs'] = runs_str
        # Parse individual runs
        runs = [r.strip() for r in runs_str.split(',') if r.strip()]
        # Filter out non-run entries (like "simulatio")
        runs = [r for r in runs if re.match(r'^R\d+$', r)]
        metadata['num_runs'] = len(runs)
        # Get the last run (usually the best/longest)
        metadata['last_run'] = runs[-1] if runs else 'R01'
    else:
        # No available_runs field found, assume R01 exists
        metadata['num_runs'] = 1
        metadata['last_run'] = 'R01'
    
    # For THC simulations, check metadata.txt format
    if 'evolution_time_M' in text:
        time_match = re.search(r'evolution_time_M\s*=\s*([\d.e+-]+)', text)
        if time_match:
            metadata['evolution_time'] = float(time_match.group(1))
    
    # Check if gravitational wave data might be available
    if 'id_gw_frequency' in text or 'gravitational' in text.lower():
        metadata['has_gw'] = True
    else:
        metadata['has_gw'] = False
        
    return metadata

def fetch_metadata(sim_type, sim_id, max_retries=3):
    """Fetch metadata_main.txt for a given simulation."""
    sim_name = f"{sim_type}_{sim_id:04d}"
    
    # Check if metadata file already exists locally
    local_path = os.path.join('CoRe', sim_name, 'metadata_main.txt')
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    print(f"Using cached metadata for {sim_name}")
                    return sim_id, content
        except Exception as e:
            print(f"Error reading cached metadata for {sim_name}: {e}")
    
    # Fetch from remote if not cached
    url = f"https://core-gitlfs.tpi.uni-jena.de/core_database/{sim_name}/-/raw/master/metadata_main.txt?ref_type=heads"
    
    for retry in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                # Save metadata locally
                try:
                    sim_dir = os.path.join('CoRe', sim_name)
                    os.makedirs(sim_dir, exist_ok=True)
                    with open(os.path.join(sim_dir, 'metadata_main.txt'), 'w') as f:
                        f.write(content)
                except Exception as e:
                    print(f"Error saving metadata for {sim_name}: {e}")
                
                return sim_id, content
            elif response.status_code == 404:
                return sim_id, None
            else:
                print(f"Error {response.status_code} for {sim_name}")
        except Exception as e:
            if retry == max_retries - 1:
                print(f"Failed to fetch {sim_name}: {e}")
        time.sleep(0.5)  # Be polite to the server
    
    return sim_id, None

def evaluate_simulation_quality(metadata):
    """Score simulation based on quality criteria for f2 extraction."""
    score = 0
    reasons = []
    
    # First check: Is it a BNS?
    is_bns = False
    if metadata.get('binary_type') == 'BNS':
        is_bns = True
        score += 10
        reasons.append("BNS merger")
    elif 'mass1' in metadata and 'mass2' in metadata:
        # Check if masses are in NS range (1.0 - 2.0 M_sun)
        if 1.0 <= metadata['mass1'] <= 2.0 and 1.0 <= metadata['mass2'] <= 2.0:
            is_bns = True
            score += 8
            reasons.append("NS mass range")
    
    # If it's a BNS, give it a baseline score to include it
    if is_bns:
        # Ensure all BNS simulations get included by giving a minimum score
        if score < 15:
            score = 15
            reasons.append("All BNS included")
    
    # Continue with quality scoring (still useful for sorting)
    # Check EoS consistency
    if 'eos' in metadata:
        score += 3
        reasons.append(f"EoS: {metadata['eos']}")
        # Known NS EoS that are good for f2 studies
        known_good_eos = ['SLy', 'SLy4', 'APR4', 'H4', 'MS1', 'ALF2', 'ENG', 'MPA1', 'DD2', 'SFHo', 'MS1b']
        if any(eos in metadata.get('eos', '') for eos in known_good_eos):
            score += 3
            reasons.append("Known good EoS")
    
    # Check if Lambda values are available
    if 'Lambda' in metadata or ('Lambda1' in metadata and 'Lambda2' in metadata):
        score += 3
        reasons.append("Lambda available")
    
    # Check eccentricity (lower is better)
    if 'eccentricity' in metadata:
        if metadata['eccentricity'] < 0.01:
            score += 3
            reasons.append("Low eccentricity")
        elif metadata['eccentricity'] < 0.02:
            score += 1
            reasons.append("Acceptable eccentricity")
    
    # Check number of runs (more runs = better statistics)
    if metadata.get('num_runs', 0) >= 3:
        score += 3
        reasons.append(f"{metadata['num_runs']} runs")
    elif metadata.get('num_runs', 0) >= 2:
        score += 1
        reasons.append(f"{metadata['num_runs']} runs")
    
    # Check evolution time (need > 10 ms post-merger for THC)
    if metadata.get('evolution_time', 0) > 5000:  # in M units, roughly > 25 ms
        score += 5
        reasons.append("Long evolution")
    elif metadata.get('evolution_time', 0) > 3000:
        score += 3
        reasons.append("Adequate evolution")
    
    # Check for GW extraction
    if metadata.get('has_gw'):
        score += 3
        reasons.append("GW data")
    
    # Special bonus for equal mass mergers (easier to calibrate)
    if 'mass1' in metadata and 'mass2' in metadata:
        q = min(metadata['mass1'], metadata['mass2']) / max(metadata['mass1'], metadata['mass2'])
        if q > 0.95:
            score += 2
            reasons.append("Equal mass")
    
    return score, reasons

def main():
    """Main function to select best simulations."""
    print("Selecting best CoRe simulations for f2 extraction...")
    print()
    
    # Create CoRe directory
    os.makedirs('CoRe', exist_ok=True)
    
    # We'll check a subset first to test
    bam_range = range(1, 227)  # BAM:0001 to BAM:0226
    thc_range = range(1, 123)  # THC:0001 to THC:0122
    
    all_results = []
    
    # Process BAM simulations
    print("\nProcessing BAM simulations...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_metadata, 'BAM', i): i for i in bam_range}
        
        for future in as_completed(futures):
            sim_id, content = future.result()
            if content:
                try:
                    metadata = parse_metadata(content)
                    metadata['simulation'] = f'BAM:{sim_id:04d}'
                    score, reasons = evaluate_simulation_quality(metadata)
                    metadata['quality_score'] = score
                    metadata['quality_reasons'] = '; '.join(reasons)
                    all_results.append(metadata)
                except Exception as e:
                    print(f"Error parsing BAM:{sim_id:04d}: {e}")
    
    # Process THC simulations
    print("\nProcessing THC simulations...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_metadata, 'THC', i): i for i in thc_range}
        
        for future in as_completed(futures):
            sim_id, content = future.result()
            if content:
                try:
                    metadata = parse_metadata(content)
                    metadata['simulation'] = f'THC:{sim_id:04d}'
                    score, reasons = evaluate_simulation_quality(metadata)
                    metadata['quality_score'] = score
                    metadata['quality_reasons'] = '; '.join(reasons)
                    all_results.append(metadata)
                except Exception as e:
                    print(f"Error parsing THC:{sim_id:04d}: {e}")
    
    # Convert to DataFrame and sort by quality score
    df = pd.DataFrame(all_results)
    df = df.sort_values('quality_score', ascending=False)
    
    # Save full results
    df.to_csv('./data/core_simulations_quality.csv', index=False)
    print(f"\nSaved {len(df)} simulations to ./data/core_simulations_quality.csv")

    best_sims = df[df['quality_score'] >= 15]
    print(f"\nFound {len(best_sims)} BNS simulations (all included):")
    
    # Display top 20
    print("\nTop 20 simulations:")
    # Select only available columns
    display_cols = ['simulation', 'quality_score', 'quality_reasons']
    if 'mass1' in df.columns:
        display_cols.insert(1, 'mass1')
    if 'mass2' in df.columns:
        display_cols.insert(2, 'mass2')
    if 'eos' in df.columns:
        display_cols.insert(3, 'eos')
    elif 'eos1' in df.columns:
        display_cols.insert(3, 'eos1')
    
    print(best_sims.head(20)[display_cols])
    
    # Save best simulations
    best_sims.to_csv('./data/best_core_simulations.csv', index=False)
    print(f"\nSaved {len(best_sims)} best simulations to ./data/best_core_simulations.csv")
    
    # Create a list of simulations to download HDF5 data from
    download_cols = ['simulation']
    if 'mass1' in best_sims.columns:
        download_cols.append('mass1')
    if 'mass2' in best_sims.columns:
        download_cols.append('mass2')
    if 'eos' in best_sims.columns:
        download_cols.append('eos')
    elif 'eos1' in best_sims.columns:
        download_cols.append('eos1')
    if 'Lambda' in best_sims.columns:
        download_cols.append('Lambda')
    if 'last_run' in best_sims.columns:
        download_cols.append('last_run')
    
    download_list = best_sims[download_cols].copy()
    # Add download URL for each simulation
    def create_url(row):
        sim_name = row['simulation'].replace(':', '_')
        # Use last_run if available, otherwise default to R01
        run = row.get('last_run', 'R01')
        if pd.isna(run) or run == '' or run is None:
            run = 'R01'
        return f"https://core-gitlfs.tpi.uni-jena.de/core_database/{sim_name}/-/raw/master/{run}/data.h5?ref_type=heads&inline=false"
    
    download_list['data_url'] = download_list.apply(create_url, axis=1)
    download_list.to_csv('./data/core_simulations_to_download.csv', index=False)
    print(f"\nCreated download list with {len(download_list)} simulations: ./data/core_simulations_to_download.csv")

if __name__ == "__main__":
    main()