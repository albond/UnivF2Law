#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
data_prep_04_download_all_catalogs.py - Master script to build a comprehensive simulation catalog.

This script:
1. Downloads simulation data from the CoRe catalog via direct JSON links.
2. Downloads simulation data from the SACRA catalog.
3. Merges both catalogs into a single master list.
4. Matches the master list against the local f2 frequency database.
5. Outputs the final, enriched dataset for calibration.
"""

import pandas as pd
import requests
import json
import numpy as np
import os
from bs4 import BeautifulSoup

# --- CoRe Catalog Functions ---

def download_and_parse_core():
    """Downloads and parses the CoRe catalog from direct JSON URLs."""
    print("--- Downloading CoRe Catalog ---")
    urls = {
        'BNS': 'https://core-gitlfs.tpi.uni-jena.de/core_database/core_database_index/-/raw/master/json/DB_NR.json?ref_type=heads',
        'BHNS': 'https://core-gitlfs.tpi.uni-jena.de/core_database/core_database_index/-/raw/master/json/DB_Hyb.json?ref_type=heads'
    }
    
    all_core_data = []

    for name, url in urls.items():
        print(f"Downloading {name} data from {url}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Could not download CoRe {name} data: {e}")
            continue
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse JSON from CoRe {name} data.")
            continue

        for entry in raw_data.get('data', []):
            try:
                # Extracting lambda from the first element of the list
                lambda_a_str = entry.get('id_Lambdaell_starA', '0,0,0').split(',')[0]
                lambda_b_str = entry.get('id_Lambdaell_starB', '0,0,0').split(',')[0]

                data_dict = {
                    'eos': entry.get('id_eos'),
                    'm1': float(entry.get('id_mass_starA', 0)),
                    'm2': float(entry.get('id_mass_starB', 0)),
                    'lambda1': float(lambda_a_str),
                    'lambda2': float(lambda_b_str),
                    'source': 'CoRe',
                    'model_name': entry.get('database_key')
                }
                if data_dict['m1'] > 0 and data_dict['m2'] > 0:
                    all_core_data.append(data_dict)
            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Could not parse CoRe entry {entry.get('database_key')}: {e}")
                continue

    if not all_core_data:
        print("ERROR: No data could be parsed from CoRe.")
        return None

    df = pd.DataFrame(all_core_data)
    df.to_csv('./data/core_catalog.csv', index=False)
    print(f"Successfully parsed and saved {len(df)} simulations from CoRe to ./data/core_catalog.csv")
    return df

# --- SACRA Catalog Functions ---

def download_sacra_catalog():
    """Download and parse SACRA catalog from the official website."""
    print("\n--- Downloading SACRA Catalog ---")
    url = "https://www2.yukawa.kyoto-u.ac.jp/~nr_kyoto/SACRA_PUB/catalog_list.html"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR downloading SACRA catalog: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    if not table:
        print("ERROR: Could not find data table in SACRA HTML")
        return None

    rows = table.find_all('tr')
    data_list = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) < 13: continue
        try:
            data_dict = {
                'model_name': cols[0].text.strip(),
                'm1': float(cols[1].text.strip()),
                'm2': float(cols[2].text.strip()),
                'eos': cols[7].text.strip(),
                'lambda1': float(cols[8].text.strip()),
                'lambda2': float(cols[9].text.strip()),
                'source': 'SACRA'
            }
            data_list.append(data_dict)
        except (ValueError, IndexError):
            continue

    if not data_list:
        print("ERROR: No valid data rows found in SACRA catalog.")
        return None

    df = pd.DataFrame(data_list)
    df.to_csv('./data/sacra_catalog.csv', index=False)
    print(f"Successfully parsed and saved {len(df)} simulations from SACRA to ./data/sacra_catalog.csv")
    return df

# --- Matching Function ---

def match_catalogs_with_f2(master_catalog_df):
    """Matches the master catalog against the local f2 database."""
    print("\n--- Matching Catalogs with F2 Data ---")
    f2_db_path = './data/integrated_f2_database.csv'
    if not os.path.exists(f2_db_path):
        print(f"ERROR: {f2_db_path} not found. Please run integrated_real_f2_data.py first.")
        return False

    f2_db = pd.read_csv(f2_db_path)

    # Extended EOS name mapping to handle all variants
    EOS_NAME_MAP = {
        # Common variations
        'APR': 'APR4', 'APR4_EPP': 'APR4', 'APR4-EPP': 'APR4',
        'SLy': 'SLy4', 'Sly': 'SLy4', 'SLY': 'SLy4', 'SLY4': 'SLy4',
        'MS1b': 'MS1', 'MS1B': 'MS1', 'ms1b': 'MS1',
        # H4 family
        '15H': 'H4', '125H': 'H4', 'HB': 'H4', 'B': 'H4', 'H4-soft': 'H4',
        # G2 variants
        'GS2': 'G2', 'G2k123': 'G2', 'G2K123': 'G2',
        # SFHo variants
        'SFX': 'SFHo', 'SFHO': 'SFHo', 'sfho': 'SFHo',
        # Other common EoS
        '2H': '2H', '2h': '2H',
        'ALF2': 'ALF2', 'alf2': 'ALF2',
        'BHBlp': 'BHBlp', 'BHBLp': 'BHBlp', 'BHBLP': 'BHBlp',
        'BLh': 'BLh', 'BLH': 'BLh',
        'BLQ': 'BLQ', 'blq': 'BLQ',
        'DD2': 'DD2', 'dd2': 'DD2',
        'ENG': 'ENG', 'eng': 'ENG',
        'LS220': 'LS220', 'ls220': 'LS220',
        'MPA1': 'MPA1', 'mpa1': 'MPA1',
        # BAM specific variations
        'BAMx': 'BA', 'BAMX': 'BA', 'BA': 'BA',
        # Additional variants from v2 paper
        'M0': 'M0', 'm0': 'M0',
        'NRPMw': 'NRPMw', 'NRPMW': 'NRPMw', 'nrpmw': 'NRPMw'
    }
    f2_db['eos_normalized'] = f2_db['eos'].apply(lambda x: EOS_NAME_MAP.get(x, x))
    master_catalog_df['eos_normalized'] = master_catalog_df['eos'].apply(lambda x: EOS_NAME_MAP.get(x, x))

    # Extract simulation name from note column for direct matching
    def extract_sim_name(note):
        if pd.isna(note):
            return None
        # Handle different formats: "CoRe BAM:0001", "SACRA BAM:0001", etc.
        parts = note.split()
        if len(parts) >= 2:
            return parts[-1]  # Return the last part (e.g., "BAM:0001")
        return note
    
    f2_db['sim_name'] = f2_db['note'].apply(extract_sim_name)
    
    # First try direct matching by simulation name
    # Include quality columns if they exist
    f2_columns = ['sim_name', 'f2_Hz', 'f2_err_Hz', 'source', 'eos_normalized', 'm1', 'm2']
    if 'quality_flag' in f2_db.columns:
        f2_columns.extend(['quality_flag', 'peak_to_median', 'n_sigma', 'pass_strict_filter'])
    
    direct_matches = pd.merge(
        master_catalog_df,
        f2_db[f2_db['sim_name'].notna()][f2_columns],
        left_on='model_name',
        right_on='sim_name',
        suffixes=('_cat', '_f2'))
    
    # For remaining f2 entries without direct match, use EOS+mass matching
    unmatched_f2 = f2_db[~f2_db.index.isin(direct_matches.index)]
    
    if len(unmatched_f2) > 0:
        # Deduplicate catalog for mass-based matching
        catalog_dedup = master_catalog_df[~master_catalog_df['model_name'].isin(direct_matches['model_name'])]
        catalog_dedup = catalog_dedup.sort_values(['lambda1', 'lambda2'], ascending=False).drop_duplicates(
            subset=['eos_normalized', 'm1', 'm2'], keep='first')
        
        # Mass-based matching for remaining entries
        # Use same columns as direct matches
        mass_matches = pd.merge(
            catalog_dedup,
            unmatched_f2[f2_columns],
            on='eos_normalized',
            suffixes=('_cat', '_f2'))
        
        # Filter for mass proximity with more tolerant threshold
        # Use 0.02-0.03 M⊙ as recommended
        mass_tol = 0.025  # 2.5% of solar mass
        mass_matches = mass_matches[np.isclose(mass_matches['m1_cat'], mass_matches['m1_f2'], atol=mass_tol) &
                                    np.isclose(mass_matches['m2_cat'], mass_matches['m2_f2'], atol=mass_tol)]
        
        # Combine both match types
        final_matches = pd.concat([direct_matches, mass_matches], ignore_index=True)
    else:
        final_matches = direct_matches
    
    print(f"Direct matches by simulation name: {len(direct_matches)}")
    print(f"Additional mass-based matches: {len(final_matches) - len(direct_matches)}")

    if final_matches.empty:
        print("ERROR: No matching simulations found after merging.")
        return False

    # Format the final dataframe
    # Check which columns actually exist
    available_columns = final_matches.columns.tolist()
    print(f"Available columns in final_matches: {available_columns}")
    
    # Build columns list dynamically based on what's available
    columns_to_select = []
    rename_dict = {}
    
    # Model name
    if 'model_name' in available_columns:
        columns_to_select.append('model_name')
    
    # Mass columns
    if 'm1_cat' in available_columns:
        columns_to_select.extend(['m1_cat', 'm2_cat'])
        rename_dict.update({'m1_cat': 'm1', 'm2_cat': 'm2'})
    elif 'm1_f2' in available_columns:
        columns_to_select.extend(['m1_f2', 'm2_f2'])
        rename_dict.update({'m1_f2': 'm1', 'm2_f2': 'm2'})
    
    # EOS column
    if 'eos_cat' in available_columns:
        columns_to_select.append('eos_cat')
        rename_dict['eos_cat'] = 'eos'
    elif 'eos' in available_columns:
        columns_to_select.append('eos')
    
    # Lambda columns
    if 'lambda1' in available_columns:
        columns_to_select.extend(['lambda1', 'lambda2'])
    
    # f2 columns
    if 'f2_Hz' in available_columns:
        columns_to_select.extend(['f2_Hz', 'f2_err_Hz'])
        rename_dict.update({'f2_Hz': 'f2_NR', 'f2_err_Hz': 'f2_err'})
    
    # Source column
    if 'source_cat' in available_columns:
        columns_to_select.append('source_cat')
        rename_dict['source_cat'] = 'catalog_source'
    elif 'source' in available_columns:
        columns_to_select.append('source')
        rename_dict['source'] = 'catalog_source'
    
    # Also get the f2 source if available
    if 'source_f2' in available_columns:
        columns_to_select.append('source_f2')
    
    # Add quality columns if they exist
    quality_cols = ['quality_flag', 'peak_to_median', 'n_sigma', 'pass_strict_filter']
    for col in quality_cols:
        if col in available_columns:
            columns_to_select.append(col)
    
    output_df = final_matches[columns_to_select].rename(columns=rename_dict)
    
    # Add detailed f2_source column
    # First check if we already have f2_source from the matching
    if 'source_f2' in final_matches.columns:
        # Use the existing f2_source from the integrated database
        output_df['f2_source'] = final_matches['source_f2']
    else:
        # Map source information to be more specific
        def get_f2_source(row):
            if 'CoRe' in str(row.get('catalog_source', '')):
                return 'CoRe_FFT_v2'  # v2 indicates improved extraction
            elif 'SACRA' in str(row.get('catalog_source', '')):
                return 'SACRA_FFT_v2'
            else:
                return 'Unknown_FFT_v2'
        
        output_df['f2_source'] = output_df.apply(get_f2_source, axis=1)
    
    # No need to drop duplicates since we're matching directly by simulation name
    # output_df = output_df.drop_duplicates(subset=['f2_NR', 'model_name'], keep='first')

    print(f"Total f2 entries in database: {len(f2_db)}")
    print(f"Successfully matched simulations: {len(output_df)}")
    print(f"Unmatched simulations: {len(f2_db) - len(output_df)}")

    # Apply quality filter - remove entries with extreme uncertainties
    print(f"\nApplying quality filter...")
    initial_count = len(output_df)
    
    # If quality flag exists, report statistics
    if 'quality_flag' in output_df.columns:
        print(f"Quality flag distribution:")
        print(f"  Flag 0 (high quality): {len(output_df[output_df['quality_flag'] == 0])}")
        print(f"  Flag 1 (weak peak): {len(output_df[output_df['quality_flag'] == 1])}")
        print(f"  Flag 2 (multi-peak): {len(output_df[output_df['quality_flag'] == 2])}")
        print(f"  Flag 3 (both issues): {len(output_df[output_df['quality_flag'] == 3])}")
        print(f"  Pass strict filter: {len(output_df[output_df['pass_strict_filter'] == True])}")
    
    # Still apply basic error filter to remove extreme outliers
    output_df = output_df[output_df['f2_err'] < 3000]  # Remove f2_err > 3000 Hz
    filtered_count = initial_count - len(output_df)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} entries with f2_err > 3000 Hz")
    
    output_file = './data/nr_simulations_with_f2.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(output_df)} quality-filtered entries to {output_file}")
    return True


def main():
    """Main pipeline function."""
    # Step 1: Download and parse CoRe data
    core_df = download_and_parse_core()
    if core_df is None:
        core_df = pd.DataFrame()

    # Step 2: Download and parse SACRA data
    sacra_df = download_sacra_catalog()
    if sacra_df is None:
        sacra_df = pd.DataFrame()

    # Step 3: Combine catalogs
    if core_df.empty and sacra_df.empty:
        print("\nERROR: Failed to download data from all sources. Cannot proceed.")
        return
    
    master_catalog = pd.concat([core_df, sacra_df], ignore_index=True)
    master_catalog.to_csv('./data/master_simulation_catalog.csv', index=False)
    print(f"\nCreated master catalog with {len(master_catalog)} total entries.")

    # Step 4: Match with f2 data
    match_catalogs_with_f2(master_catalog)

if __name__ == "__main__":
    main()
