#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Ablation analysis of M2 model parameters
Tests the contribution of each parameter by fixing or removing them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os
import emcee
from config import (ERROR_SCALING_FACTOR, INITIAL_PARAMS_M2_FIXED_W,
                    calculate_aic_bic_likelihood, predict_f2)

# Physical constants
G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
c = 299_792_458  # Speed of light [m/s]
M_sun = 1.98847e30  # Solar mass [kg]

def compute_tilde_lambda(m1, m2, lambda1, lambda2):
    """Compute effective tidal deformability"""
    M_tot = m1 + m2
    return (16/13) * ((m1 + 12*m2) * m1**4 * lambda1 + 
                      (m2 + 12*m1) * m2**4 * lambda2) / M_tot**5

# Full M2 model
def f_model_M2_full(x, alpha, beta, delta, lambda0, w):
    """Full M2 model with all parameters"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    return f_model * (1.0 + delta_eos)

# Ablated models
def f_model_M2_fixed_w(x, alpha, beta, delta, lambda0, w_fixed=0.1):
    """M2 with fixed width parameter"""
    f_base, q, tilde_lambda = x
    return predict_f2(f_base, q, tilde_lambda, alpha, beta, delta, lambda0, w=w_fixed)

def f_model_M2_no_eos(x, alpha, beta):
    """M2 without EOS correction (equivalent to M0)"""
    f_base, q, tilde_lambda = x
    return alpha * f_base * (1.0 + beta * (q - 1)**2)

def f_model_M2_step(x, alpha, beta, delta_soft, delta_hard, lambda_crit):
    """M2 with step function instead of logistic"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    delta_eos = np.where(np.log10(tilde_lambda) < lambda_crit, delta_soft, delta_hard)
    return f_model * (1.0 + delta_eos)

def f_model_M2_linear_eos(x, alpha, beta, delta_slope):
    """M2 with linear EOS correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    delta_eos = delta_slope * (np.log10(tilde_lambda) - 3.0)
    return f_model * (1.0 + delta_eos)


def run_mcmc_for_model(f_obs, f_err, x_data, model_func, p0, param_names, bounds):
    """Run MCMC analysis with proper convergence"""
    ndim = len(p0)
    nwalkers = 2 * ndim * (ndim + 1)
    
    def log_likelihood(theta):
        try:
            f_pred = model_func(x_data, *theta)
            residuals = (f_obs - f_pred) / f_err
            # Add small regularization to avoid numerical issues
            chi2 = np.sum(residuals**2)
            # Add log of normalizing constant for proper likelihood
            log_norm = -0.5 * len(f_obs) * np.log(2 * np.pi) - np.sum(np.log(f_err))
            return -0.5 * chi2 + log_norm
        except Exception as e:
            print(f"Error in likelihood: {e}")
            return -np.inf
    
    def log_prior(theta):
        for i, (val, bound) in enumerate(zip(theta, bounds)):
            if not (bound[0] <= val <= bound[1]):
                return -np.inf
        return 0.0
    
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    # Initialize walkers with proper scatter (1% of parameter value)
    pos = []
    for i in range(nwalkers):
        while True:
            # Start from best-fit with 1% perturbation
            trial = p0.copy()
            for j in range(ndim):
                scale = 0.01 * max(abs(p0[j]), 0.1)
                trial[j] += scale * np.random.randn()
            # Check bounds
            if all(bounds[j][0] <= trial[j] <= bounds[j][1] for j in range(ndim)):
                pos.append(trial)
                break
    pos = np.array(pos)
    
    # Run MCMC with convergence monitoring
    # Use moves with different scales for better exploration
    moves = [
        (emcee.moves.StretchMove(a=2.0), 0.8),
        (emcee.moves.StretchMove(a=1.5), 0.2),
    ]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=moves)
    
    # Initial test for acceptance fraction
    test_steps = 100
    pos, prob, state = sampler.run_mcmc(pos, test_steps, progress=False)
    initial_acc = np.mean(sampler.acceptance_fraction)
    
    # Adjust walker initialization if acceptance is too low
    if initial_acc < 0.1:
        sampler.reset()
        # Reinitialize with larger scatter for lower acceptance
        pos = []
        scatter_scale = 0.05  # 5% scatter for wider exploration
        
        for i in range(nwalkers):
            while True:
                trial = p0.copy()
                for j in range(ndim):
                    scale = scatter_scale * max(abs(p0[j]), 0.1)
                    trial[j] += scale * np.random.randn()
                
                if all(bounds[j][0] <= trial[j] <= bounds[j][1] for j in range(ndim)):
                    pos.append(trial)
                    break
        pos = np.array(pos)
    
    # Continue burn-in
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, 1000, progress=False)
    sampler.reset()
    
    # Production run - shorter for ablation analysis
    sampler.run_mcmc(pos, 2000, progress=False)
    
    # Get samples with appropriate thinning
    try:
        tau = sampler.get_autocorr_time()
        thin = int(np.max(tau) / 2)
        discard = int(2 * np.max(tau))
    except:
        thin = 15
        discard = 1000
    
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    # Calculate statistics
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    medians = percentiles[1]
    
    return medians, samples, sampler

def analyze_data_subgroups(df, data_dict):
    """Analyze model performance on data subgroups"""
    results = {}
    
    # Group by EOS
    eos_column = 'eos' if 'eos' in df.columns else None
    if eos_column:
        eos_groups = df.groupby(eos_column)
        results['by_eos'] = {}
        
        for eos, group in eos_groups:
            if len(group) < 10:  # Skip small groups
                continue
                
            # Prepare data for this group
            # Use boolean mask instead of indices to handle filtered data
            mask = df[eos_column] == eos
            x_data = (data_dict['f_base'][mask], 
                      data_dict['q'][mask], 
                      data_dict['tilde_Lambda'][mask])
            f_obs = data_dict['f_obs'][mask]
            f_err = data_dict['f_err'][mask]
            
            # Fit model
            try:
                # Convert bounds to tuple format for curve_fit
                bounds_tuple = ([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1])
                popt, _ = curve_fit(f_model_M2_full, x_data, f_obs, 
                                   p0=INITIAL_PARAMS_M2_FIXED_W + [0.1],  # Use reference params
                                   sigma=f_err, absolute_sigma=True,
                                   bounds=bounds_tuple)
                
                f_pred = f_model_M2_full(x_data, *popt)
                chi2_dof = np.sum(((f_obs - f_pred) / f_err)**2) / (len(f_obs) - 5)
                
                results['by_eos'][eos] = {
                    'n_points': len(group),
                    'chi2_dof': chi2_dof,
                    'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt))
                }
            except:
                continue
    
    # Group by mass ratio
    q_bins = [0.7, 0.85, 0.95, 1.01]
    results['by_q'] = {}
    
    for i in range(len(q_bins)-1):
        mask = (data_dict['q'] >= q_bins[i]) & (data_dict['q'] < q_bins[i+1])
        if np.sum(mask) < 10:
            continue
            
        x_data = (data_dict['f_base'][mask], 
                  data_dict['q'][mask], 
                  data_dict['tilde_Lambda'][mask])
        f_obs = data_dict['f_obs'][mask]
        f_err = data_dict['f_err'][mask]
        
        try:
            # Convert bounds to tuple format for curve_fit
            bounds_tuple = ([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1])
            popt, _ = curve_fit(f_model_M2_full, x_data, f_obs,
                               p0=INITIAL_PARAMS_M2_FIXED_W + [0.1],  # Use reference params
                               sigma=f_err, absolute_sigma=True,
                               bounds=bounds_tuple)
            
            f_pred = f_model_M2_full(x_data, *popt)
            chi2_dof = np.sum(((f_obs - f_pred) / f_err)**2) / (len(f_obs) - 5)
            
            results['by_q'][f'{q_bins[i]:.2f}-{q_bins[i+1]:.2f}'] = {
                'n_points': np.sum(mask),
                'chi2_dof': chi2_dof,
                'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt))
            }
        except:
            continue
    
    return results

def main():
    """Main ablation analysis workflow"""
    # Load data
    data_file = "./data/nr_simulations_with_f2.csv"
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Quality filtering
    if 'f2_err' in df.columns:
        error_threshold = df['f2_err'].quantile(0.95)
        df = df[(df['f2_err'] < error_threshold) & (df['f2_err'] > 0)]
    
    # Prepare data
    m1 = df[['m1', 'm2']].max(axis=1).values
    m2 = df[['m1', 'm2']].min(axis=1).values
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    
    q = m2 / m1
    M_tot = m1 + m2
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)
    
    R_L = tilde_Lambda**(1/5) * (G * (M_tot * M_sun)) / c**2
    f_base = np.sqrt(G * (M_tot * M_sun) / R_L**3) / (2 * np.pi)
    
    f_obs = df['f2_NR'].values
    f_err_raw = df.get('f2_err', np.full(len(df), 50.0)).values
    error_scaling = ERROR_SCALING_FACTOR  # From config.py: 0.997
    f_err = f_err_raw * error_scaling
    
    x_data = (f_base, q, tilde_Lambda)
    
    data_dict = {
        'f_base': f_base, 'q': q, 'tilde_Lambda': tilde_Lambda,
        'f_obs': f_obs, 'f_err': f_err
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("ABLATION ANALYSIS OF M2 MODEL")
    print("="*80)
    
    # 1. Full M2 model (baseline)
    print("\n1. Full M2 model (5 parameters)")
    bounds_full = list(zip([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1]))
    popt_full, samples_full, _ = run_mcmc_for_model(
        f_obs, f_err, x_data, f_model_M2_full,
        p0=INITIAL_PARAMS_M2_FIXED_W + [0.1],  # Use reference params + w
        param_names=['alpha', 'beta', 'delta', 'lambda0', 'w'],
        bounds=bounds_full
    )
    
    f_pred_full = f_model_M2_full(x_data, *popt_full)
    aic_full, bic_full, chi2_full = calculate_aic_bic_likelihood(f_obs, f_pred_full, f_err, 5)
    
    results['M2_full'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt_full)),
        'aic': aic_full, 'bic': bic_full, 'chi2': chi2_full,
        'chi2_dof': chi2_full / (len(f_obs) - 5)
    }
    
    # 2. Fixed w at different values
    for w_fixed in [0.05, 0.1, 0.2]:
        print(f"\n2. M2 with fixed w = {w_fixed}")
        
        def model_fixed_w(x, alpha, beta, delta, lambda0):
            return f_model_M2_fixed_w(x, alpha, beta, delta, lambda0, w_fixed)
        
        bounds_fixed_w = list(zip([0.1, -10, -1, 2], [10, 10, 1, 3.5]))
        popt_fixed_w, _, _ = run_mcmc_for_model(
            f_obs, f_err, x_data, model_fixed_w,
            p0=INITIAL_PARAMS_M2_FIXED_W,  # Use reference params
            param_names=['alpha', 'beta', 'delta', 'lambda0'],
            bounds=bounds_fixed_w
        )
        
        f_pred_fixed_w = model_fixed_w(x_data, *popt_fixed_w)
        aic, bic, chi2 = calculate_aic_bic_likelihood(f_obs, f_pred_fixed_w, f_err, 4)
        
        results[f'M2_w_fixed_{w_fixed}'] = {
            'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0'], popt_fixed_w)),
            'w_fixed': w_fixed,
            'aic': aic, 'bic': bic, 'chi2': chi2,
            'chi2_dof': chi2 / (len(f_obs) - 4)
        }
    
    # 3. No EOS correction (M0)
    print("\n3. M2 without EOS correction (M0)")
    bounds_no_eos = list(zip([0.1, -10], [10, 10]))
    popt_no_eos, _, _ = run_mcmc_for_model(
        f_obs, f_err, x_data, f_model_M2_no_eos,
        p0=[4.5, -0.8],
        param_names=['alpha', 'beta'],
        bounds=bounds_no_eos
    )
    
    f_pred_no_eos = f_model_M2_no_eos(x_data, *popt_no_eos)
    aic, bic, chi2 = calculate_aic_bic_likelihood(f_obs, f_pred_no_eos, f_err, 2)
    
    results['M2_no_eos'] = {
        'params': dict(zip(['alpha', 'beta'], popt_no_eos)),
        'aic': aic, 'bic': bic, 'chi2': chi2,
        'chi2_dof': chi2 / (len(f_obs) - 2)
    }
    
    # 4. Linear EOS correction
    print("\n4. M2 with linear EOS correction")
    bounds_linear = list(zip([0.1, -10, -1], [10, 10, 1]))
    popt_linear, _, _ = run_mcmc_for_model(
        f_obs, f_err, x_data, f_model_M2_linear_eos,
        p0=[4.5, -0.8, -0.1],
        param_names=['alpha', 'beta', 'delta_slope'],
        bounds=bounds_linear
    )
    
    f_pred_linear = f_model_M2_linear_eos(x_data, *popt_linear)
    aic, bic, chi2 = calculate_aic_bic_likelihood(f_obs, f_pred_linear, f_err, 3)
    
    results['M2_linear_eos'] = {
        'params': dict(zip(['alpha', 'beta', 'delta_slope'], popt_linear)),
        'aic': aic, 'bic': bic, 'chi2': chi2,
        'chi2_dof': chi2 / (len(f_obs) - 3)
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'k':<3} {'χ²/dof':<8} {'ΔAIC':<10} {'ΔBIC':<10}")
    print("-"*80)
    
    min_aic = min(r['aic'] for r in results.values())
    min_bic = min(r['bic'] for r in results.values())
    
    for model_name, r in results.items():
        n_params = len(r['params'])
        if 'w_fixed' in r:
            n_params += 1  # Count fixed parameter
        delta_aic = r['aic'] - min_aic
        delta_bic = r['bic'] - min_bic
        print(f"{model_name:<20} {n_params:<3} {r['chi2_dof']:<8.3f} "
              f"{delta_aic:<10.1f} {delta_bic:<10.1f}")
    
    # Analyze parameter contributions
    print("\n" + "="*80)
    print("PARAMETER CONTRIBUTION ANALYSIS")
    print("="*80)
    
    # Calculate relative importance of each parameter
    chi2_full = results['M2_full']['chi2']
    chi2_no_eos = results['M2_no_eos']['chi2']
    chi2_linear = results['M2_linear_eos']['chi2']
    chi2_w_fixed = results['M2_w_fixed_0.1']['chi2']
    
    print(f"Improvement from adding EOS correction: Δχ² = {chi2_no_eos - chi2_full:.1f}")
    print(f"Loss from linear vs logistic EOS: Δχ² = {chi2_linear - chi2_full:.1f}")
    print(f"Loss from fixing w = 0.1: Δχ² = {chi2_w_fixed - chi2_full:.1f}")
    
    # Analyze subgroups
    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS")
    print("="*80)
    
    subgroup_results = analyze_data_subgroups(df, data_dict)
    
    print("\nBy EOS:")
    for eos, res in subgroup_results['by_eos'].items():
        print(f"  {eos}: n={res['n_points']}, χ²/dof={res['chi2_dof']:.3f}, "
              f"λ₀={res['params']['lambda0']:.3f}")
    
    print("\nBy mass ratio:")
    for q_range, res in subgroup_results['by_q'].items():
        print(f"  q∈[{q_range}]: n={res['n_points']}, χ²/dof={res['chi2_dof']:.3f}, "
              f"δ={res['params']['delta']:.3f}")
    
    # Save results
    output_dir = './results'
    
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
    
    serializable_results = convert_to_serializable({
        'ablation_results': results,
        'subgroup_analysis': subgroup_results
    })
    
    with open(os.path.join(output_dir, 'ablation_analysis.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Model comparison
    models = list(results.keys())
    chi2_values = [r['chi2_dof'] for r in results.values()]
    
    ax1.bar(range(len(models)), chi2_values)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('χ²/dof')
    ax1.set_title('Ablation Analysis: Model Fit Quality')
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5)
    
    # Panel 2: Parameter variation across EOS
    eos_names = list(subgroup_results['by_eos'].keys())
    lambda0_values = [res['params']['lambda0'] for res in subgroup_results['by_eos'].values()]
    
    ax2.scatter(range(len(eos_names)), lambda0_values, s=100)
    ax2.set_xticks(range(len(eos_names)))
    ax2.set_xticklabels(eos_names, rotation=45, ha='right')
    ax2.set_ylabel('λ₀')
    ax2.set_title('λ₀ Variation Across EOS')
    ax2.axhline(np.mean(lambda0_values), color='r', linestyle='--', 
                label=f'Mean = {np.mean(lambda0_values):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('./figs', 'fig_ablation_analysis.png'), dpi=300)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()