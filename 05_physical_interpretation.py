#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Physical interpretation of M2 model parameters
Analyzes simulations near λ₀ threshold to understand physical meaning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
from config import M2_FIXED_W_PARAMS, ERROR_SCALING_FACTOR, LAMBDA0_VALUE

# Physical constants
G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
c = 299_792_458  # Speed of light [m/s]
M_sun = 1.98847e30  # Solar mass [kg]

def compute_tilde_lambda(m1, m2, lambda1, lambda2):
    """Compute effective tidal deformability"""
    M_tot = m1 + m2
    return (16/13) * ((m1 + 12*m2) * m1**4 * lambda1 + 
                      (m2 + 12*m1) * m2**4 * lambda2) / M_tot**5

def compute_m_threshold(tilde_lambda):
    """Compute threshold mass for collapse to BH"""
    return 2.38 + 3.606e-4 * tilde_lambda**0.858

def analyze_threshold_physics(df, lambda0=LAMBDA0_VALUE):
    """Analyze physical differences above and below λ₀ threshold"""
    # Prepare data
    m1 = df[['m1', 'm2']].max(axis=1).values
    m2 = df[['m1', 'm2']].min(axis=1).values
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    
    M_tot = m1 + m2
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)
    
    # Divide into groups
    soft_mask = tilde_Lambda < lambda0
    hard_mask = tilde_Lambda >= lambda0
    
    results = {
        'soft_eos': {},
        'hard_eos': {},
        'transition': {}
    }
    
    # Analyze collapse threshold proximity
    m_thresh = compute_m_threshold(tilde_Lambda)
    proximity_to_collapse = M_tot / m_thresh
    
    results['soft_eos']['mean_proximity'] = np.mean(proximity_to_collapse[soft_mask])
    results['hard_eos']['mean_proximity'] = np.mean(proximity_to_collapse[hard_mask])
    
    # Check for remnant lifetime if available
    if 't_remnant' in df.columns:
        results['soft_eos']['mean_lifetime_ms'] = np.mean(df.loc[soft_mask, 't_remnant'])
        results['hard_eos']['mean_lifetime_ms'] = np.mean(df.loc[hard_mask, 't_remnant'])
    
    # Analyze frequency characteristics
    results['soft_eos']['mean_f2'] = np.mean(df.loc[soft_mask, 'f2_NR'])
    results['hard_eos']['mean_f2'] = np.mean(df.loc[hard_mask, 'f2_NR'])
    results['soft_eos']['std_f2'] = np.std(df.loc[soft_mask, 'f2_NR'])
    results['hard_eos']['std_f2'] = np.std(df.loc[hard_mask, 'f2_NR'])
    
    # Check for secondary peaks
    if 'f_secondary' in df.columns:
        results['soft_eos']['secondary_peak_fraction'] = np.mean(df.loc[soft_mask, 'f_secondary'] > 0)
        results['hard_eos']['secondary_peak_fraction'] = np.mean(df.loc[hard_mask, 'f_secondary'] > 0)
    
    # Analyze transition region (λ₀ ± 10%)
    transition_mask = (tilde_Lambda > 0.9 * lambda0) & (tilde_Lambda < 1.1 * lambda0)
    results['transition']['n_simulations'] = np.sum(transition_mask)
    results['transition']['f2_variance'] = np.var(df.loc[transition_mask, 'f2_NR'])
    
    return results, proximity_to_collapse

def test_physical_hypotheses(df, params):
    """Test specific physical hypotheses for parameters"""
    lambda0 = 10**params['parameters']['lambda0']
    # Handle both M2 and M2_fixed_w models
    w = params['parameters'].get('w', 0.1)  # Default to 0.1 if w not in parameters
    delta = params['parameters']['delta']
    
    print("\n" + "="*80)
    print("TESTING PHYSICAL HYPOTHESES")
    print("="*80)
    
    # Hypothesis 1: λ₀ separates prompt vs delayed collapse
    print("\nHypothesis 1: λ₀ = {:.0f} separates collapse regimes".format(lambda0))
    
    threshold_analysis, proximity = analyze_threshold_physics(df, lambda0)
    
    print("  Soft EOS (Λ̃ < λ₀):")
    print(f"    Mean proximity to collapse: {threshold_analysis['soft_eos']['mean_proximity']:.3f}")
    print(f"    Mean f_2: {threshold_analysis['soft_eos']['mean_f2']:.0f} Hz")
    
    print("  Hard EOS (Λ̃ ≥ λ₀):")
    print(f"    Mean proximity to collapse: {threshold_analysis['hard_eos']['mean_proximity']:.3f}")
    print(f"    Mean f_2: {threshold_analysis['hard_eos']['mean_f2']:.0f} Hz")
    
    # Statistical test
    m1 = df[['m1', 'm2']].max(axis=1).values
    m2 = df[['m1', 'm2']].min(axis=1).values
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)
    
    soft_proximity = proximity[tilde_Lambda < lambda0]
    hard_proximity = proximity[tilde_Lambda >= lambda0]
    
    t_stat, p_value = stats.ttest_ind(soft_proximity, hard_proximity)
    print(f"  T-test for proximity difference: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Hypothesis 2: w reflects EOS diversity
    print(f"\nHypothesis 2: w = {w:.3f} reflects EOS transition sharpness")
    
    # Analyze frequency scatter in transition region
    transition_width = w * np.log(10) * lambda0  # Convert from log-space width
    transition_min = lambda0 * 10**(-w)
    transition_max = lambda0 * 10**(w)
    
    transition_mask = (tilde_Lambda > transition_min) & (tilde_Lambda < transition_max)
    n_transition = np.sum(transition_mask)
    
    if n_transition > 5:
        f2_scatter_transition = np.std(df.loc[transition_mask, 'f2_NR'])
        f2_scatter_overall = np.std(df['f2_NR'])
        
        print(f"  Transition region: Λ̃ ∈ [{transition_min:.0f}, {transition_max:.0f}]")
        print(f"  N simulations in transition: {n_transition}")
        print(f"  f_2 scatter in transition: {f2_scatter_transition:.0f} Hz")
        print(f"  f_2 scatter overall: {f2_scatter_overall:.0f} Hz")
        print(f"  Ratio: {f2_scatter_transition/f2_scatter_overall:.2f}")
    
    # Hypothesis 3: δ reflects energy loss mechanisms
    print(f"\nHypothesis 3: δ = {delta:.3f} reflects additional dissipation")
    
    # Calculate effective compactness
    M_tot = m1 + m2
    R_L = tilde_Lambda**(1/5) * (G * (M_tot * M_sun)) / c**2
    compactness = (G * M_tot * M_sun) / (R_L * c**2)
    
    # Correlation between compactness and model residuals
    f_base = np.sqrt(G * (M_tot * M_sun) / R_L**3) / (2 * np.pi)
    q = m2 / m1
    
    # Simple model without EOS correction
    f_simple = params['parameters']['alpha'] * f_base * (1.0 + params['parameters']['beta'] * (q - 1)**2)
    residual_ratio = df['f2_NR'] / f_simple
    
    # Analyze correlation
    soft_mask = tilde_Lambda < lambda0
    correlation = np.corrcoef(compactness[soft_mask], residual_ratio[soft_mask])[0, 1]
    
    print(f"  Correlation between compactness and frequency deficit (soft EOS): {correlation:.3f}")
    print(f"  Mean frequency reduction for soft EOS: {(1 - np.mean(residual_ratio[soft_mask]))*100:.1f}%")
    print(f"  Theoretical δ effect: {abs(delta)*100:.1f}%")
    
    return threshold_analysis

def plot_physical_interpretation(df, params, output_dir):
    """Create plots for physical interpretation"""
    # Prepare data
    m1 = df[['m1', 'm2']].max(axis=1).values
    m2 = df[['m1', 'm2']].min(axis=1).values
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    
    M_tot = m1 + m2
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)
    lambda0 = 10**params['parameters']['lambda0']
    
    # Calculate proximity to collapse
    m_thresh = compute_m_threshold(tilde_Lambda)
    proximity = M_tot / m_thresh
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Proximity to collapse vs tilde Lambda
    ax1 = axes[0, 0]
    scatter = ax1.scatter(tilde_Lambda, proximity, c=df['f2_NR'], 
                         cmap='viridis', alpha=0.6, s=50)
    ax1.axvline(lambda0, color='red', linestyle='--', lw=2, 
                label=f'λ₀ = {lambda0:.0f}')
    ax1.axhline(1.0, color='black', linestyle=':', alpha=0.5,
                label='Collapse threshold')
    ax1.set_xlabel('$\\tilde{\\Lambda}$')
    ax1.set_ylabel('$M_{tot} / M_{threshold}$')
    ax1.set_xscale('log')
    ax1.set_title('Proximity to Collapse')
    ax1.legend()
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('$f_2$ [Hz]')
    
    # Panel 2: Frequency distribution by regime
    ax2 = axes[0, 1]
    soft_f2 = df.loc[tilde_Lambda < lambda0, 'f2_NR']
    hard_f2 = df.loc[tilde_Lambda >= lambda0, 'f2_NR']
    
    ax2.hist(soft_f2, bins=20, alpha=0.5, label=f'Soft (Λ̃ < {lambda0:.0f})', 
             density=True, color='red')
    ax2.hist(hard_f2, bins=20, alpha=0.5, label=f'Hard (Λ̃ ≥ {lambda0:.0f})', 
             density=True, color='blue')
    ax2.set_xlabel('$f_2$ [Hz]')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Frequency Distribution by EOS Regime')
    ax2.legend()
    
    # Panel 3: Compactness analysis
    ax3 = axes[1, 0]
    R_L = tilde_Lambda**(1/5) * (G * (M_tot * M_sun)) / c**2
    compactness = (G * M_tot * M_sun) / (R_L * c**2)
    
    ax3.scatter(compactness, df['f2_NR'], c=tilde_Lambda, 
               cmap='coolwarm', alpha=0.6, s=50)
    ax3.set_xlabel('Compactness $GM/Rc^2$')
    ax3.set_ylabel('$f_2$ [Hz]')
    ax3.set_title('Frequency vs Compactness')
    cbar3 = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=ax3)
    cbar3.set_label('$\\tilde{\\Lambda}$')
    
    # Panel 4: Transition region analysis
    ax4 = axes[1, 1]
    
    # Define narrow bins around lambda0
    lambda_bins = np.logspace(np.log10(lambda0*0.5), np.log10(lambda0*2.0), 20)
    bin_centers = (lambda_bins[:-1] + lambda_bins[1:]) / 2
    
    f2_mean = []
    f2_std = []
    
    for i in range(len(lambda_bins)-1):
        mask = (tilde_Lambda >= lambda_bins[i]) & (tilde_Lambda < lambda_bins[i+1])
        if np.sum(mask) > 3:
            f2_mean.append(np.mean(df.loc[mask, 'f2_NR']))
            f2_std.append(np.std(df.loc[mask, 'f2_NR']))
        else:
            f2_mean.append(np.nan)
            f2_std.append(np.nan)
    
    ax4.errorbar(bin_centers, f2_mean, yerr=f2_std, fmt='o-', capsize=5)
    ax4.axvline(lambda0, color='red', linestyle='--', lw=2, 
                label=f'λ₀ = {lambda0:.0f}')
    ax4.set_xlabel('$\\tilde{\\Lambda}$')
    ax4.set_ylabel('$f_2$ [Hz]')
    ax4.set_xscale('log')
    ax4.set_title('Frequency Behavior Near Transition')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_physical_interpretation.png'), dpi=300)
    plt.close()
    
    # Additional plot: Parameter correlations
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate EOS correction for each point
    # Handle both M2 and M2_fixed_w models
    w_value = params['parameters'].get('w', 0.1)  # Default to 0.1 if w not in parameters
    delta_eos = params['parameters']['delta'] / (
        1.0 + np.exp((np.log10(tilde_Lambda) - params['parameters']['lambda0']) / 
                     w_value)
    )
    
    # Color by EOS type if available
    if 'eos_name' in df.columns:
        eos_types = df['eos_name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(eos_types)))
        
        for i, eos in enumerate(eos_types):
            mask = df['eos_name'] == eos
            ax.scatter(tilde_Lambda[mask], delta_eos[mask] * 100, 
                      label=eos, alpha=0.7, s=50, color=colors[i])
    else:
        ax.scatter(tilde_Lambda, delta_eos * 100, alpha=0.6, s=50)
    
    ax.axvline(lambda0, color='red', linestyle='--', lw=2, alpha=0.7)
    ax.set_xlabel('$\\tilde{\\Lambda}$')
    ax.set_ylabel('EOS Correction [%]')
    ax.set_xscale('log')
    ax.set_title('EOS Correction Distribution')
    if 'eos_name' in df.columns:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_eos_correction_by_type.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main workflow for physical interpretation"""
    # Load data and parameters
    data_file = "./data/nr_simulations_with_f2.csv"
    params_file = "./results/optimal_model.json"
    
    df = pd.read_csv(data_file)
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Override with reference parameters if M2_fixed_w
    if params.get('model_type') == 'M2_fixed_w':
        params['parameters'] = M2_FIXED_W_PARAMS.copy()
    
    # Quality filtering
    if 'f2_err' in df.columns:
        error_threshold = df['f2_err'].quantile(0.95)
        df = df[(df['f2_err'] < error_threshold) & (df['f2_err'] > 0)]
    
    # Run physical interpretation analysis
    threshold_analysis = test_physical_hypotheses(df, params)
    
    # Create visualizations
    output_dir = './figs'
    plot_physical_interpretation(df, params, output_dir)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    # Save analysis results
    results_dir = './results'
    with open(os.path.join(results_dir, 'physical_interpretation.json'), 'w') as f:
        json.dump(convert_to_serializable(threshold_analysis), f, indent=2)
    
    print("\nPhysical interpretation analysis complete!")

if __name__ == "__main__":
    main()