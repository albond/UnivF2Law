#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
download_sacra_data.py - Download and parse SACRA catalog data

This script:
1. Downloads the SACRA catalog HTML page
2. Parses neutron star merger simulation parameters
3. Creates a template CSV for adding f2 frequency data
4. Provides instructions for completing the dataset

The SACRA catalog contains simulation parameters but NOT f2 frequencies.
Those must be extracted from the referenced papers manually.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

def download_sacra_catalog():
    """Download and parse SACRA catalog from the official website."""
    url = "https://www2.yukawa.kyoto-u.ac.jp/~nr_kyoto/SACRA_PUB/catalog_list.html"
    
    print("Downloading SACRA catalog from:")
    print(url)
    print()
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print("Successfully downloaded catalog page")
    except requests.exceptions.RequestException as e:
        print(f"ERROR downloading catalog: {e}")
        return None
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table - SACRA uses a simple table structure
    table = soup.find('table')
    if not table:
        print("ERROR: Could not find data table in HTML")
        return None
    
    # Extract data rows
    rows = table.find_all('tr')
    print(f"Found {len(rows)} rows in table")
    
    # Parse data
    data_list = []
    for i, row in enumerate(rows[1:], 1):  # Skip header
        cols = row.find_all('td')
        if len(cols) < 13:
            continue
            
        try:
            # Extract simulation parameters
            model_name = cols[0].text.strip()
            m1 = float(cols[1].text.strip())
            m2 = float(cols[2].text.strip())
            m_total = float(cols[3].text.strip())
            q = float(cols[4].text.strip())
            eta = float(cols[5].text.strip())
            m_chirp = float(cols[6].text.strip())
            eos = cols[7].text.strip()
            lambda1 = float(cols[8].text.strip())
            lambda2 = float(cols[9].text.strip())
            lambda_tilde = float(cols[10].text.strip())
            
            # Extract reference link
            ref_link = ""
            ref_cell = cols[13]
            if ref_cell.find('a'):
                ref_link = ref_cell.find('a')['href']
            
            data_dict = {
                'model_name': model_name,
                'm1': m1,
                'm2': m2,
                'm_total': m_total,
                'q': q,
                'eta': eta,
                'm_chirp': m_chirp,
                'eos': eos,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'lambda_tilde': lambda_tilde,
                'reference': ref_link
            }
            
            data_list.append(data_dict)
            
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Warning: Could not parse row {i}: {e}")
            continue
    
    if not data_list:
        print("ERROR: No valid data rows found")
        return None
    
    df = pd.DataFrame(data_list)
    print(f"\nSuccessfully parsed {len(df)} simulations")
    
    # Print summary statistics
    print("\nData summary:")
    print(f"  EoS types: {', '.join(df['eos'].unique())}")
    print(f"  Mass range: [{df['m1'].min():.2f}, {df['m1'].max():.2f}] M_sun")
    print(f"  q range: [{df['q'].min():.3f}, {df['q'].max():.3f}]")
    print(f"  Lambda_tilde range: [{df['lambda_tilde'].min():.0f}, {df['lambda_tilde'].max():.0f}]")
    
    return df

def create_f2_data_template(sacra_df):
    """Create a template CSV file for adding f2 frequency data."""
    
    # Add columns for f2 data
    sacra_df['f2_NR'] = np.nan  # To be filled from papers
    sacra_df['f2_err'] = 50     # Default 50 Hz uncertainty
    sacra_df['f2_source'] = ''   # Paper/figure where f2 was found
    sacra_df['notes'] = ''       # Additional notes
    
    # Group by reference to help with data entry
    ref_groups = sacra_df.groupby('reference').size()
    print(f"\nReference summary ({len(ref_groups)} unique papers):")
    for ref, count in ref_groups.items():
        if ref:
            print(f"  {ref}: {count} simulations")
    
    # Save template
    template_file = './data/sacra_f2_template.csv'
    sacra_df.to_csv(template_file, index=False)
    print(f"\nCreated template file: {template_file}")
    
    return template_file

def match_sacra_with_f2(sacra_df):
    """
    Match SACRA simulations with f2 data from integrated database,
    normalizing EoS names for better matching.
    """
    
    if not os.path.exists('./data/integrated_f2_database.csv'):
        print("ERROR: ./data/integrated_f2_database.csv not found.")
        print("Please run integrated_real_f2_data.py first.")
        return False
    
    # Read f2 database
    f2_db = pd.read_csv('./data/integrated_f2_database.csv')
    
    # --- EoS Normalization Map ---
    # Maps names from literature (f2_db) to SACRA canonical names
    EOS_NAME_MAP = {
        'APR': 'APR4',
        'SLy': 'SLy4',
        'MS1b': 'MS1',
        '15H': 'H4',
        '125H': 'H4',
        'HB': 'H4',
        'B': 'H4',
        'ENG': 'ENG',
        'MPA1': 'MPA1',
        'ALF2': 'ALF2',
        'GS2': 'GS2',
        'DD2': 'DD2',
        'SFX': 'SFX',
        'TM1': 'TM1',
        'SFHo': 'SFHo',
        'NRPMw': 'NRPMw',
        'LR': 'LR',
        'M0': 'M0',
        'SR': 'SR'
    }
    
    # Normalize EoS names in the f2 database
    f2_db['eos_normalized'] = f2_db['eos'].apply(lambda x: EOS_NAME_MAP.get(x, x))
    
    # --- Matching Logic ---
    matched_data = []
    unmatched_count = 0
    
    # Get unique EoS names from SACRA for better error reporting
    sacra_eos_names = set(sacra_df['eos'].unique())
    
    for _, f2_row in f2_db.iterrows():
        normalized_eos = f2_row['eos_normalized']
        
        if normalized_eos not in sacra_eos_names:
            unmatched_count += 1
            continue

        # Find SACRA simulations with matching EoS and masses
        matches = sacra_df[
            (sacra_df['eos'] == normalized_eos) &
            (np.isclose(sacra_df['m1'], f2_row['m1'], atol=0.01)) &
            (np.isclose(sacra_df['m2'], f2_row['m2'], atol=0.01))
        ]
        
        if not matches.empty:
            sacra_row = matches.iloc[0]
            
            matched_data.append({
                'model_name': sacra_row['model_name'],
                'm1': sacra_row['m1'],
                'm2': sacra_row['m2'],
                'eos': sacra_row['eos'],
                'lambda1': sacra_row['lambda1'],
                'lambda2': sacra_row['lambda2'],
                'lambda_tilde': sacra_row['lambda_tilde'],
                'f2_NR': f2_row['f2_Hz'],
                'f2_err': f2_row.get('f2_err_Hz', 50),
                'f2_source': f2_row['source'],
                'reference': sacra_row['reference']
            })
        else:
            unmatched_count += 1

    print("\n--- Matching Diagnostics ---")
    if not matched_data:
        print("ERROR: No matching simulations found after normalization.")
        print(f"Total f2 entries processed: {len(f2_db)}")
        print(f"Total unmatched entries: {unmatched_count}")
        return False
        
    result = pd.DataFrame(matched_data)
    
    result = result.drop_duplicates(subset=['eos', 'm1', 'm2', 'f2_NR'])
    
    print(f"Total f2 entries in database: {len(f2_db)}")
    print(f"Successfully matched simulations: {len(result)}")
    print(f"Unmatched simulations: {len(f2_db) - len(result)}")
    print("--------------------------")

    output_file = './data/nr_simulations_with_f2.csv'
    result.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved matched data to {output_file}")
    
    return True

def print_f2_extraction_guide():
    """Print instructions for extracting f2 data from papers."""
    
    print("\n" + "="*70)
    print("INSTRUCTIONS FOR COMPLETING F2 DATA")
    print("="*70)
    
    print("""
The SACRA catalog does not include post-merger f2 frequencies.
These must be extracted from the referenced papers manually.

Steps to complete the data:

1. Open sacra_f2_template.csv in a spreadsheet program

2. For each simulation, check the 'reference' column for the paper link

3. In each paper, look for:
   - Tables with post-merger frequencies (usually in kHz)
   - Figures showing frequency spectra with labeled peaks
   - Sections discussing "f2", "post-merger", or "gravitational wave frequency"

4. Common papers with f2 data:
   - Bauswein et al. 2012 (arXiv:1204.1888): Table with f_peak values
   - Takami et al. 2014 (arXiv:1403.5672): f2 frequencies in figures/tables
   - Takami et al. 2015 (arXiv:1412.3240): Comprehensive f2 data
   - Kiuchi et al. papers: Often show spectrograms with f2 peaks

5. Fill in the f2_NR column with the frequency in Hz (not kHz!)
   - If paper gives kHz, multiply by 1000
   - If multiple modes, use the dominant f2 mode

6. Update f2_source with paper section (e.g., "Table 2", "Fig. 5")

7. Add any relevant notes (e.g., "uncertain due to low amplitude")

8. Save the completed file as: nr_simulations_with_f2.csv

9. Run calibrate_final_model.py again

Example f2 values for common EoS:
- SLy4 (M=2.7): ~2.8-3.2 kHz
- APR4 (M=2.7): ~3.0-3.4 kHz
- H4 (M=2.7): ~2.4-2.8 kHz
""")
    print("="*70 + "\n")

def main():
    """Main function to download and process SACRA data."""
    
    print("SACRA Data Download Tool")
    print("========================\n")
    
    # Check if we already have the raw data
    if os.path.exists('./data/sacra_catalog_raw.csv'):
        print("Found existing ./data/sacra_catalog_raw.csv")
        sacra_df = pd.read_csv('./data/sacra_catalog_raw.csv')
        print(f"Loaded {len(sacra_df)} simulations from cache")
        # Always ensure sacra_catalog.csv exists
        if not os.path.exists('./data/sacra_catalog.csv'):
            sacra_df.to_csv('./data/sacra_catalog.csv', index=False)
            print("Created ./data/sacra_catalog.csv from cached data")
    else:
        print("No cached data found. Downloading from SACRA website...")
        sacra_df = download_sacra_catalog()
        if sacra_df is not None:
            sacra_df.to_csv('./data/sacra_catalog_raw.csv', index=False)
            sacra_df.to_csv('./data/sacra_catalog.csv', index=False)
            print("Saved catalog to ./data/sacra_catalog_raw.csv and ./data/sacra_catalog.csv")
    
    if sacra_df is None:
        print("ERROR: Could not download SACRA catalog")
        return
    
    # Create template for f2 data
    template_file = create_f2_data_template(sacra_df)
    
    # Print instructions
    print_f2_extraction_guide()
    
    # Try to match with f2 database if it exists
    if os.path.exists('./data/integrated_f2_database.csv'):
        print("\nFound integrated f2 database. Attempting to match with SACRA data...")
        if match_sacra_with_f2(sacra_df):
            print("\nSuccessfully created nr_simulations_with_f2.csv!")
            print("You can now run calibrate_final_model.py")
        else:
            print("\nMatching failed. Please check that both data files exist.")
    else:
        print("\nNo ./data/integrated_f2_database.csv found.")
        print("Run integrated_real_f2_data.py first to create f2 database.")
        print("\nAlternatively, fill in f2 data manually:")
        print(f"Template location: {os.path.abspath(template_file)}")

if __name__ == "__main__":
    main()