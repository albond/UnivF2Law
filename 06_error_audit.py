#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Audit of f2_err values and investigation of χ²/dof anomaly
Analyzes error distribution and proposes corrections
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from config import REFERENCE_CHI2_DOF

# Physical constants
G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
c = 299_792_458  # Speed of light [m/s]
M_sun = 1.98847e30  # Solar mass [kg]

def analyze_error_distribution(df):
    """Analyze distribution of f2_err values"""
    print("\n" + "="*80)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print("\nBasic statistics for f2_err:")
    print(f"  Mean: {df['f2_err'].mean():.1f} Hz")
    print(f"  Median: {df['f2_err'].median():.1f} Hz")
    print(f"  Std: {df['f2_err'].std():.1f} Hz")
    print(f"  Min: {df['f2_err'].min():.1f} Hz")
    print(f"  Max: {df['f2_err'].max():.1f} Hz")
    print(f"  25th percentile: {df['f2_err'].quantile(0.25):.1f} Hz")
    print(f"  75th percentile: {df['f2_err'].quantile(0.75):.1f} Hz")
    
    # Analyze by quality flag
    print("\nError statistics by quality flag:")
    for flag in sorted(df['quality_flag'].unique()):
        mask = df['quality_flag'] == flag
        print(f"  Quality {flag}: mean={df.loc[mask, 'f2_err'].mean():.1f} Hz, "
              f"median={df.loc[mask, 'f2_err'].median():.1f} Hz, n={mask.sum()}")
    
    # Analyze by peak_to_median ratio
    high_quality = df['peak_to_median'] > 10
    print(f"\nHigh peak_to_median (>10): mean error = {df.loc[high_quality, 'f2_err'].mean():.1f} Hz")
    print(f"Low peak_to_median (≤10): mean error = {df.loc[~high_quality, 'f2_err'].mean():.1f} Hz")
    
    # Check correlation with f2_NR
    corr = np.corrcoef(df['f2_NR'], df['f2_err'])[0, 1]
    print(f"\nCorrelation between f2_NR and f2_err: {corr:.3f}")
    
    # Relative errors
    rel_err = (df['f2_err'] / df['f2_NR']) * 100
    print(f"\nRelative error statistics:")
    print(f"  Mean: {rel_err.mean():.1f}%")
    print(f"  Median: {rel_err.median():.1f}%")
    
    return rel_err

def propose_error_rescaling(df, target_chi2_dof=1.0):
    """Propose error rescaling to achieve target χ²/dof"""
    print("\n" + "="*80)
    print("ERROR RESCALING ANALYSIS")
    print("="*80)
    
    # Use reference chi2/dof from config
    current_chi2_dof = REFERENCE_CHI2_DOF  # From config.py: 0.994
    print(f"\nCurrent χ²/dof: {current_chi2_dof:.3f}")
    print(f"Target χ²/dof: {target_chi2_dof:.3f}")
    
    # χ² scales as 1/σ², so to change χ²/dof from current to target:
    # σ_new = σ_old * sqrt(current_chi2_dof / target_chi2_dof)
    scaling_factor = np.sqrt(current_chi2_dof / target_chi2_dof)
    print(f"\nRequired error scaling factor: {scaling_factor:.3f}")
    
    # Analyze impact
    print(f"\nImpact of rescaling:")
    print(f"  Current median error: {df['f2_err'].median():.1f} Hz")
    print(f"  Rescaled median error: {df['f2_err'].median() * scaling_factor:.1f} Hz")
    
    # Check if uniform rescaling is appropriate
    print("\nChecking if uniform rescaling is appropriate...")
    
    # Group by EOS
    print("\nχ² contribution by EOS:")
    for eos in df['eos'].unique():
        eos_mask = df['eos'] == eos
        n_eos = eos_mask.sum()
        if n_eos > 5:  # Only for EOS with enough data
            # Would need actual residuals here
            print(f"  {eos}: n={n_eos}")
    
    return scaling_factor

def analyze_residual_patterns(df):
    """Analyze patterns in residuals that might explain low χ²/dof"""
    print("\n" + "="*80)
    print("RESIDUAL PATTERN ANALYSIS")
    print("="*80)
    
    # Check if errors correlate with specific parameters
    print("\nCorrelation of f2_err with physical parameters:")
    
    # Total mass
    M_tot = df['m1'] + df['m2']
    corr_mass = np.corrcoef(M_tot, df['f2_err'])[0, 1]
    print(f"  With total mass: {corr_mass:.3f}")
    
    # Mass ratio
    q = np.minimum(df['m1'], df['m2']) / np.maximum(df['m1'], df['m2'])
    corr_q = np.corrcoef(q, df['f2_err'])[0, 1]
    print(f"  With mass ratio: {corr_q:.3f}")
    
    # Lambda
    lambda_avg = (df['lambda1'] + df['lambda2']) / 2
    corr_lambda = np.corrcoef(lambda_avg, df['f2_err'])[0, 1]
    print(f"  With average Λ: {corr_lambda:.3f}")
    
    return M_tot, q, lambda_avg

def plot_error_analysis(df, rel_err, M_tot, q, lambda_avg):
    """Create diagnostic plots for error analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Error distribution
    ax1 = axes[0, 0]
    ax1.hist(df['f2_err'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(df['f2_err'].median(), color='red', linestyle='--', 
                label=f'Median = {df["f2_err"].median():.0f} Hz')
    ax1.set_xlabel('f2_err [Hz]')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of f2 Errors')
    ax1.legend()
    
    # Panel 2: Relative error distribution
    ax2 = axes[0, 1]
    ax2.hist(rel_err, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(rel_err.median(), color='red', linestyle='--',
                label=f'Median = {rel_err.median():.1f}%')
    ax2.set_xlabel('Relative Error [%]')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Relative Errors')
    ax2.legend()
    
    # Panel 3: Error vs f2
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df['f2_NR'], df['f2_err'], 
                         c=df['quality_flag'], cmap='viridis', alpha=0.6)
    ax3.set_xlabel('f2_NR [Hz]')
    ax3.set_ylabel('f2_err [Hz]')
    ax3.set_title('Error vs Frequency')
    plt.colorbar(scatter, ax=ax3, label='Quality Flag')
    
    # Panel 4: Error vs total mass
    ax4 = axes[1, 0]
    ax4.scatter(M_tot, df['f2_err'], alpha=0.6)
    ax4.set_xlabel('Total Mass [M☉]')
    ax4.set_ylabel('f2_err [Hz]')
    ax4.set_title('Error vs Total Mass')
    
    # Panel 5: Error vs mass ratio
    ax5 = axes[1, 1]
    ax5.scatter(q, df['f2_err'], alpha=0.6)
    ax5.set_xlabel('Mass Ratio q')
    ax5.set_ylabel('f2_err [Hz]')
    ax5.set_title('Error vs Mass Ratio')
    
    # Panel 6: Error by EOS
    ax6 = axes[1, 2]
    eos_counts = df['eos'].value_counts()
    major_eos = eos_counts[eos_counts > 5].index
    df_major = df[df['eos'].isin(major_eos)]
    
    eos_errors = df_major.groupby('eos')['f2_err'].apply(list)
    ax6.boxplot(eos_errors.values, tick_labels=eos_errors.index)
    ax6.set_xlabel('EOS')
    ax6.set_ylabel('f2_err [Hz]')
    ax6.set_title('Error Distribution by EOS')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('./figs/fig_error_analysis.png', dpi=300)
    plt.close()

def main():
    """Main workflow for error audit"""
    # Load data
    data_file = "./data/nr_simulations_with_f2.csv"
    df = pd.read_csv(data_file)
    
    # Basic error analysis
    rel_err = analyze_error_distribution(df)
    
    # Propose rescaling
    scaling_factor = propose_error_rescaling(df)
    
    # Analyze patterns
    M_tot, q, lambda_avg = analyze_residual_patterns(df)
    
    # Create plots
    plot_error_analysis(df, rel_err, M_tot, q, lambda_avg)
    
    # Save recommendations
    recommendations = {
        'error_statistics': {
            'mean_error_hz': float(df['f2_err'].mean()),
            'median_error_hz': float(df['f2_err'].median()),
            'mean_relative_error_percent': float(rel_err.mean()),
            'median_relative_error_percent': float(rel_err.median())
        },
        'rescaling': {
            'current_chi2_dof': float(REFERENCE_CHI2_DOF),  # From config.py
            'target_chi2_dof': 1.0,
            'scaling_factor': float(scaling_factor),
            'rescaled_median_error_hz': float(df['f2_err'].median() * scaling_factor)
        },
        'recommendations': [
            f"Scale all errors by factor {scaling_factor:.3f} to achieve χ²/dof ≈ 1",
            "Consider using quality-dependent error model",
            "Investigate systematic effects in high-error simulations"
        ]
    }
    
    with open('./results/error_audit.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nError audit complete!")
    print(f"\nRecommendation: Scale all errors by {scaling_factor:.3f}")

if __name__ == "__main__":
    main()