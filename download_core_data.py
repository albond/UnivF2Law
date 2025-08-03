#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
download_core_data.py - Download and parse CoRe simulation data.

This script:
1. Uses the core-watpy library to access the CoRe database.
2. Downloads metadata for all public binary neutron star simulations.
3. Extracts key parameters (masses, EoS, Lambda).
4. Saves the data to a CSV file for later aggregation.
"""

import pandas as pd
from watpy.coredb import coredb

def download_core_catalog():
    """Download and parse the CoRe simulation catalog."""
    print("Initializing CoRe database connection...")
    try:
        db = coredb.Database(catalog='mergers/bns')
        print("Successfully connected to CoRe BNS catalog.")
    except Exception as e:
        print(f"ERROR: Could not connect to CoRe database: {e}")
        return None

    # Fetch all records
    records = db.records
    print(f"Found {len(records)} total records in the BNS catalog.")

    data_list = []
    for record in records:
        try:
            # Extract parameters
            # Some records might be missing certain keys, so we use .get()
            m1 = record.get('mass1')
            m2 = record.get('mass2')
            eos = record.get('eos')
            
            # Lambda parameters might be nested or named differently
            # We need to investigate the record structure more deeply
            # For now, let's assume a simple structure
            lambda1 = record.get('lambda1')
            lambda2 = record.get('lambda2')

            # Skip if essential data is missing
            if not all([m1, m2, eos]):
                continue

            data_dict = {
                'model_name': record.name,
                'm1': m1,
                'm2': m2,
                'eos': eos,
                'lambda1': lambda1, # Might be None
                'lambda2': lambda2, # Might be None
                'source': 'CoRe'
            }
            data_list.append(data_dict)

        except Exception as e:
            print(f"Warning: Could not parse record {record.name}: {e}")
            continue

    if not data_list:
        print("ERROR: No valid data rows could be parsed from CoRe.")
        return None

    df = pd.DataFrame(data_list)
    print(f"Successfully parsed {len(df)} simulations from CoRe.")
    return df

def main():
    """Main function to download and save CoRe data."""
    print("CoRe Data Download Tool")
    print("=======================\n")

    core_df = download_core_catalog()

    if core_df is not None:
        output_file = './data/core_catalog.csv'
        core_df.to_csv(output_file, index=False)
        print(f"Saved CoRe catalog to {output_file}")
    else:
        print("ERROR: Failed to download CoRe catalog.")

if __name__ == "__main__":
    main()
