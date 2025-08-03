#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Download HDF5 data files from CoRe database for selected simulations.
This script downloads data.h5 files and organizes them in the CoRe directory.
"""

import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def download_hdf5(simulation, url, save_dir='CoRe'):
    """Download HDF5 file for a specific simulation."""
    # Create directory for this simulation
    sim_dir = os.path.join(save_dir, simulation.replace(':', '_'))
    os.makedirs(sim_dir, exist_ok=True)
    
    # Target file path
    hdf5_path = os.path.join(sim_dir, 'data.h5')
    
    # Check if already downloaded
    if os.path.exists(hdf5_path):
        file_size = os.path.getsize(hdf5_path)
        if file_size > 1000000:  # At least 1MB
            print(f"✓ {simulation} already downloaded ({file_size/1e6:.1f} MB)")
            return True
    
    print(f"↓ Downloading {simulation}...")
    
    try:
        # Download with streaming to handle large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Write to file
        downloaded = 0
        with open(hdf5_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r  {simulation}: {progress:.1f}%", end='', flush=True)
        
        print(f"\r✓ {simulation} downloaded ({downloaded/1e6:.1f} MB)    ")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {simulation}: {e}")
        # Remove partial file
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        return False
    except Exception as e:
        print(f"✗ Error downloading {simulation}: {e}")
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        return False

def main():
    """Main function to download HDF5 files."""
    # Load the best simulations list
    if not os.path.exists('./data/best_core_simulations.csv'):
        print("Error: ./data/best_core_simulations.csv not found!")
        print("Run data_prep_01_select_core_simulations.py first.")
        return
    
    # Read all best simulations
    best_sims = pd.read_csv('./data/best_core_simulations.csv')
    print(f"Found {len(best_sims)} best simulations")
    
    # Prepare download list with necessary columns
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
    
    df = best_sims[download_cols].copy()
    
    # Add download URLs
    df['data_url'] = df.apply(
        lambda row: f"https://core-gitlfs.tpi.uni-jena.de/core_database/{row['simulation'].replace(':', '_')}/-/raw/master/{row.get('last_run', 'R01')}/data.h5?ref_type=heads&inline=false",
        axis=1
    )
    
    # Filter out already downloaded simulations
    already_downloaded = []
    for _, row in df.iterrows():
        sim_dir = os.path.join('CoRe', row['simulation'].replace(':', '_'))
        hdf5_path = os.path.join(sim_dir, 'data.h5')
        if os.path.exists(hdf5_path) and os.path.getsize(hdf5_path) > 0:
            already_downloaded.append(row['simulation'])
    
    print(f"Already downloaded: {len(already_downloaded)} simulations")
    
    # Remove already downloaded from the list
    df = df[~df['simulation'].isin(already_downloaded)]
    print(f"Need to download: {len(df)} simulations")
    
    # Create CoRe directory
    os.makedirs('CoRe', exist_ok=True)
    
    # Download statistics
    successful = 0
    failed = []
    
    # Download in parallel (but limited to avoid overwhelming the server)
    max_workers = 3  # Be polite to the server
    
    print("\nStarting downloads...")
    print("=" * 60)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_sim = {}
        for _, row in df.iterrows():
            future = executor.submit(download_hdf5, row['simulation'], row['data_url'])
            future_to_sim[future] = row['simulation']
        
        # Process completed downloads
        for future in as_completed(future_to_sim):
            sim = future_to_sim[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed.append(sim)
            except Exception as e:
                print(f"✗ Exception for {sim}: {e}")
                failed.append(sim)
            
            # Small delay to be polite
            time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Download Summary:")
    print(f"  Successful: {successful}/{len(df)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed simulations:")
        for sim in failed:
            print(f"  - {sim}")
        
        # Save failed list for retry
        with open('failed_downloads.txt', 'w') as f:
            for sim in failed:
                f.write(f"{sim}\n")
        print("\nFailed simulations saved to failed_downloads.txt")
    
    # Create summary of downloaded data
    downloaded_sims = []
    for sim_dir in os.listdir('CoRe'):
        hdf5_path = os.path.join('CoRe', sim_dir, 'data.h5')
        if os.path.exists(hdf5_path):
            file_size = os.path.getsize(hdf5_path) / 1e6  # MB
            sim_name = sim_dir.replace('_', ':')
            # Find original data
            matching_sims = df[df['simulation'] == sim_name]
            if len(matching_sims) > 0:
                sim_data = matching_sims.iloc[0].to_dict()
                sim_data['file_size_mb'] = file_size
                downloaded_sims.append(sim_data)
            else:
                # Simulation was downloaded but not in current df (from previous run)
                downloaded_sims.append({
                    'simulation': sim_name,
                    'file_size_mb': file_size
                })
    
    if downloaded_sims:
        summary_df = pd.DataFrame(downloaded_sims)
        summary_df.to_csv('./data/downloaded_simulations.csv', index=False)
        print(f"\nCreated summary: ./data/downloaded_simulations.csv")
        print(f"Total downloaded: {len(summary_df)} simulations")
        print(f"Total size: {summary_df['file_size_mb'].sum():.1f} MB")

if __name__ == "__main__":
    main()