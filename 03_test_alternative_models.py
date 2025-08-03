#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Test alternative EOS correction forms and mass-dependent models
Implements polynomial, Padé approximants, and explicit mass dependencies
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
    """
    Compute effective tidal deformability
    
    Parameters:
    -----------
    m1, m2 : float
        Component masses in solar masses [M_sun]
    lambda1, lambda2 : float
        Dimensionless tidal deformabilities of each component
        
    Returns:
    --------
    tilde_lambda : float
        Combined dimensionless tidal deformability
    """
    M_tot = m1 + m2  # Total mass in solar masses
    return (16/13) * ((m1 + 12*m2) * m1**4 * lambda1 + 
                      (m2 + 12*m1) * m2**4 * lambda2) / M_tot**5

# Original optimal model (M2)
def f_model_M2_original(x, alpha, beta, delta, lambda0, w):
    """M2: Base model + logistic EOS correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    return f_model * (1.0 + delta_eos)

# M2 with fixed w=0.1 (the actual optimal model)
def f_model_M2_fixed_w(x, alpha, beta, delta, lambda0):
    """M2_fixed_w: Base model + logistic EOS correction with w=0.1"""
    f_base, q, tilde_lambda = x
    return predict_f2(f_base, q, tilde_lambda, alpha, beta, delta, lambda0, w=0.1)

# Alternative 1: Polynomial EOS correction
def f_model_polynomial(x, alpha, beta, delta1, delta2):
    """Model with polynomial EOS correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    log_lambda = np.log10(tilde_lambda)
    delta_eos = delta1 * (log_lambda - 3.0) + delta2 * (log_lambda - 3.0)**2
    return f_model * (1.0 + delta_eos)

# Alternative 2: Padé approximant
def f_model_pade(x, alpha, beta, a0, a1, b1):
    """Model with Padé [1/1] EOS correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    log_lambda = np.log10(tilde_lambda)
    delta_eos = (a0 + a1 * (log_lambda - 3.0)) / (1.0 + b1 * (log_lambda - 3.0))
    return f_model * (1.0 + delta_eos)

# Alternative 3: Mass-dependent model
def f_model_mass_dependent(x, alpha, beta, delta, lambda0, w, gamma):
    """Model with explicit mass dependence"""
    f_base, q, tilde_lambda, m_tot = x
    m_ref = 2.7  # Reference total mass [M_sun]
    
    # Base model with mass-dependent alpha
    alpha_eff = alpha * (1.0 + gamma * (m_tot - m_ref) / m_ref)
    f_model = alpha_eff * f_base * (1.0 + beta * (q - 1)**2)
    
    # Standard EOS correction
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    return f_model * (1.0 + delta_eos)

# Alternative 4: Step function EOS correction
def f_model_step(x, alpha, beta, delta_soft, delta_hard, lambda_crit):
    """Model with step function EOS correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # Step function
    delta_eos = np.where(tilde_lambda < 10**lambda_crit, delta_soft, delta_hard)
    return f_model * (1.0 + delta_eos)

def robust_regression(x_data, y_data, y_err, model_func, p0, bounds, method='huber'):
    """
    Perform robust regression using Huber or other robust estimators
    
    Parameters:
    -----------
    method : str
        'huber' for Huber regression, 'theil_sen' for Theil-Sen estimator
    """
    from sklearn.linear_model import HuberRegressor
    from scipy.optimize import minimize
    
    if method == 'huber':
        # Define loss function for Huber regression
        def huber_loss(params, x, y, epsilon=1.35):
            y_pred = model_func(x, *params)
            residuals = y - y_pred
            
            # Huber loss
            loss = np.where(
                np.abs(residuals) <= epsilon,
                0.5 * residuals**2,
                epsilon * (np.abs(residuals) - 0.5 * epsilon)
            )
            return np.sum(loss)
        
        # Minimize Huber loss
        result = minimize(
            huber_loss, p0, args=(x_data, y_data),
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result.x, None
    
    else:
        # Fall back to standard curve_fit
        return curve_fit(model_func, x_data, y_data, p0=p0, 
                        sigma=y_err, absolute_sigma=True, bounds=bounds)


def run_mcmc_fit(f_obs, f_err, x_data, model_func, p0, param_names, bounds):
    """Run MCMC analysis for model fitting"""
    ndim = len(p0)
    nwalkers = 2 * ndim * (ndim + 1)
    
    def log_likelihood(theta):
        try:
            f_pred = model_func(x_data, *theta)
            chi2 = np.sum(((f_obs - f_pred) / f_err)**2)
            return -0.5 * chi2
        except:
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
    
    # Initialize walkers
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Run MCMC (shorter runs for comparison)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    # Burn-in
    pos, prob, state = sampler.run_mcmc(pos, 200, progress=False)
    sampler.reset()
    
    # Production
    sampler.run_mcmc(pos, 500, progress=False)
    
    # Get samples
    samples = sampler.get_chain(discard=50, thin=5, flat=True)
    
    # Calculate statistics
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    medians = percentiles[1]
    
    return medians, samples

def test_models(df):
    """Test all alternative models"""
    # Prepare data
    m1 = df[['m1', 'm2']].max(axis=1).values  # Primary mass [M_sun]
    m2 = df[['m1', 'm2']].min(axis=1).values  # Secondary mass [M_sun]
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    
    q = m2 / m1  # Mass ratio
    M_tot = m1 + m2  # Total mass [M_sun]
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)
    
    # Calculate base frequency
    R_L = tilde_Lambda**(1/5) * (G * (M_tot * M_sun)) / c**2  # [meters]
    f_base = np.sqrt(G * (M_tot * M_sun) / R_L**3) / (2 * np.pi)  # [Hz]
    
    f_obs = df['f2_NR'].values
    f_err = df.get('f2_err', np.full(len(df), 50.0)).values * ERROR_SCALING_FACTOR
    
    results = {}
    
    print("\nTesting alternative models with MCMC...")
    
    # 1. Original M2 (for comparison)
    print("\n1. Original M2 (logistic) - MCMC fit")
    x_M2 = (f_base, q, tilde_Lambda)
    p0_M2 = INITIAL_PARAMS_M2_FIXED_W + [0.1]  # Add w=0.1 for full M2
    bounds_M2 = list(zip([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1]))
    
    popt_M2, samples_M2 = run_mcmc_fit(f_obs, f_err, x_M2, f_model_M2_original, 
                                       p0_M2, ['alpha', 'beta', 'delta', 'lambda0', 'w'], 
                                       bounds_M2)
    
    f_pred_M2 = f_model_M2_original(x_M2, *popt_M2)
    aic_M2, bic_M2, chi2_M2 = calculate_aic_bic_likelihood(f_obs, f_pred_M2, f_err, 5)
    
    results['M2_original'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt_M2)),
        'aic': aic_M2, 'bic': bic_M2, 'chi2': chi2_M2,
        'chi2_dof': chi2_M2 / (len(f_obs) - 5)
    }
    
    # 1b. M2_fixed_w (the actual optimal model)
    print("\n1b. M2_fixed_w (logistic with w=0.1) - MCMC fit")
    p0_M2_fixed = INITIAL_PARAMS_M2_FIXED_W  # Use reference parameters
    bounds_M2_fixed = list(zip([0.1, -10, -1, 2], [10, 10, 1, 3.5]))
    
    popt_M2_fixed, samples_M2_fixed = run_mcmc_fit(f_obs, f_err, x_M2, f_model_M2_fixed_w, 
                                                   p0_M2_fixed, ['alpha', 'beta', 'delta', 'lambda0'], 
                                                   bounds_M2_fixed)
    
    f_pred_M2_fixed = f_model_M2_fixed_w(x_M2, *popt_M2_fixed)
    aic_M2_fixed, bic_M2_fixed, chi2_M2_fixed = calculate_aic_bic_likelihood(f_obs, f_pred_M2_fixed, f_err, 4)
    
    results['M2_fixed_w'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0'], popt_M2_fixed)),
        'aic': aic_M2_fixed, 'bic': bic_M2_fixed, 'chi2': chi2_M2_fixed,
        'chi2_dof': chi2_M2_fixed / (len(f_obs) - 4)
    }
    
    # 2. Polynomial EOS
    print("\n2. Polynomial EOS correction - MCMC fit")
    x_poly = (f_base, q, tilde_Lambda)
    p0_poly = [4.5, -0.8, -0.1, -0.05]
    bounds_poly = list(zip([0.1, -10, -1, -1], [10, 10, 1, 1]))
    
    popt_poly, samples_poly = run_mcmc_fit(f_obs, f_err, x_poly, f_model_polynomial,
                                          p0_poly, ['alpha', 'beta', 'delta1', 'delta2'],
                                          bounds_poly)
    
    f_pred_poly = f_model_polynomial(x_poly, *popt_poly)
    aic_poly, bic_poly, chi2_poly = calculate_aic_bic_likelihood(f_obs, f_pred_poly, f_err, 4)
    
    results['polynomial'] = {
        'params': dict(zip(['alpha', 'beta', 'delta1', 'delta2'], popt_poly)),
        'aic': aic_poly, 'bic': bic_poly, 'chi2': chi2_poly,
        'chi2_dof': chi2_poly / (len(f_obs) - 4)
    }
    
    # 3. Padé approximant
    print("\n3. Padé [1/1] approximant - MCMC fit")
    x_pade = (f_base, q, tilde_Lambda)
    p0_pade = [4.5, -0.8, -0.1, -0.2, 0.5]
    bounds_pade = list(zip([0.1, -10, -1, -1, -2], [10, 10, 1, 1, 2]))
    
    popt_pade, samples_pade = run_mcmc_fit(f_obs, f_err, x_pade, f_model_pade,
                                          p0_pade, ['alpha', 'beta', 'a0', 'a1', 'b1'],
                                          bounds_pade)
    
    f_pred_pade = f_model_pade(x_pade, *popt_pade)
    aic_pade, bic_pade, chi2_pade = calculate_aic_bic_likelihood(f_obs, f_pred_pade, f_err, 5)
    
    results['pade'] = {
        'params': dict(zip(['alpha', 'beta', 'a0', 'a1', 'b1'], popt_pade)),
        'aic': aic_pade, 'bic': bic_pade, 'chi2': chi2_pade,
        'chi2_dof': chi2_pade / (len(f_obs) - 5)
    }
    
    # 4. Mass-dependent model
    print("\n4. Mass-dependent model - MCMC fit")
    x_mass = (f_base, q, tilde_Lambda, M_tot)
    p0_mass = [4.5, -0.8, -0.5, 3.1, 0.1, -0.1]
    bounds_mass = list(zip([0.1, -10, -1, 2, 0.05, -1], [10, 10, 1, 3.5, 1, 1]))
    
    popt_mass, samples_mass = run_mcmc_fit(f_obs, f_err, x_mass, f_model_mass_dependent,
                                          p0_mass, ['alpha', 'beta', 'delta', 'lambda0', 'w', 'gamma'],
                                          bounds_mass)
    
    f_pred_mass = f_model_mass_dependent(x_mass, *popt_mass)
    aic_mass, bic_mass, chi2_mass = calculate_aic_bic_likelihood(f_obs, f_pred_mass, f_err, 6)
    
    results['mass_dependent'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w', 'gamma'], popt_mass)),
        'aic': aic_mass, 'bic': bic_mass, 'chi2': chi2_mass,
        'chi2_dof': chi2_mass / (len(f_obs) - 6)
    }
    
    # 5. Step function
    print("\n5. Step function EOS - MCMC fit")
    x_step = (f_base, q, tilde_Lambda)
    p0_step = [4.5, -0.8, -0.4, -0.1, 3.1]
    bounds_step = list(zip([0.1, -10, -1, -0.5, 2.5], [10, 10, 0, 0.5, 3.5]))
    
    popt_step, samples_step = run_mcmc_fit(f_obs, f_err, x_step, f_model_step,
                                          p0_step, ['alpha', 'beta', 'delta_soft', 'delta_hard', 'lambda_crit'],
                                          bounds_step)
    
    f_pred_step = f_model_step(x_step, *popt_step)
    aic_step, bic_step, chi2_step = calculate_aic_bic_likelihood(f_obs, f_pred_step, f_err, 5)
    
    results['step'] = {
        'params': dict(zip(['alpha', 'beta', 'delta_soft', 'delta_hard', 'lambda_crit'], popt_step)),
        'aic': aic_step, 'bic': bic_step, 'chi2': chi2_step,
        'chi2_dof': chi2_step / (len(f_obs) - 5)
    }
    
    # 6. Test robust regression on original M2
    print("\n6. M2 with robust regression")
    # For robust regression, fall back to non-MCMC approach
    popt_robust, _ = robust_regression(x_M2, f_obs, f_err, f_model_M2_original, 
                                      INITIAL_PARAMS_M2_FIXED_W + [0.1], list(zip([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1])), 
                                      method='huber')
    
    f_pred_robust = f_model_M2_original(x_M2, *popt_robust)
    aic_robust, bic_robust, chi2_robust = calculate_aic_bic_likelihood(f_obs, f_pred_robust, f_err, 5)
    
    results['M2_robust'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt_robust)),
        'aic': aic_robust, 'bic': bic_robust, 'chi2': chi2_robust,
        'chi2_dof': chi2_robust / (len(f_obs) - 5)
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("ALTERNATIVE MODELS COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'k':<3} {'χ²/dof':<8} {'AIC':<10} {'BIC':<10}")
    print("-"*80)
    
    min_aic = min(r['aic'] for r in results.values())
    min_bic = min(r['bic'] for r in results.values())
    
    for model_name, r in results.items():
        n_params = len(r['params'])
        delta_aic = r['aic'] - min_aic
        delta_bic = r['bic'] - min_bic
        print(f"{model_name:<20} {n_params:<3} {r['chi2_dof']:<8.3f} "
              f"{r['aic']:<10.1f} ({delta_aic:+.1f}) "
              f"{r['bic']:<10.1f} ({delta_bic:+.1f})")
    
    # Save results
    output_dir = './results'
    with open(os.path.join(output_dir, 'alternative_models.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Test extrapolation
    print("\n" + "="*80)
    print("EXTRAPOLATION VALIDATION")
    print("="*80)
    
    # Test on extreme values
    q_test = np.array([0.7, 0.75, 0.85, 0.95, 1.0])  # Extended q range
    lambda_test = np.array([50, 100, 1000, 3000, 5000])  # Extended Lambda range
    m_tot_test = np.array([2.2, 2.4, 2.7, 3.0, 3.5])  # Extended mass range
    
    print("\nChecking model behavior at boundary conditions...")
    
    # Create test grid
    test_cases = []
    for qt in q_test:
        for lt in lambda_test:
            for mt in m_tot_test:
                # Calculate test f_base
                R_test = lt**(1/5) * (G * (mt * M_sun)) / c**2
                f_base_test = np.sqrt(G * (mt * M_sun) / R_test**3) / (2 * np.pi)
                
                # Predict with optimal model
                f_pred = f_model_M2_original(
                    (f_base_test, qt, lt),
                    *popt_M2
                )
                
                # Check for reasonable values (2000-5000 Hz typical range)
                if 1000 < f_pred < 6000:
                    status = "OK"
                else:
                    status = "WARNING"
                
                if qt < 0.8 or lt < 100 or lt > 3000 or mt < 2.4 or mt > 3.2:
                    regime = "EXTRAPOLATION"
                else:
                    regime = "interpolation"
                
                if regime == "EXTRAPOLATION" and status == "WARNING":
                    print(f"  q={qt:.2f}, Λ̃={lt:4.0f}, M={mt:.1f} M☉: "
                          f"f_2={f_pred:.0f} Hz [{status}]")
    
    return results

def plot_alternative_eos_forms(df, results):
    """Plot comparison of different EOS correction forms"""
    output_dir = './figs'
    
    # Prepare Lambda range for plotting
    lambda_range = np.logspace(1.5, 3.5, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: EOS correction profiles
    # Original logistic
    p_orig = results['M2_original']['params']
    delta_logistic = p_orig['delta'] / (1.0 + np.exp((np.log10(lambda_range) - p_orig['lambda0']) / p_orig['w']))
    
    # Polynomial
    p_poly = results['polynomial']['params']
    log_lambda = np.log10(lambda_range)
    delta_poly = p_poly['delta1'] * (log_lambda - 3.0) + p_poly['delta2'] * (log_lambda - 3.0)**2
    
    # Padé
    p_pade = results['pade']['params']
    delta_pade = (p_pade['a0'] + p_pade['a1'] * (log_lambda - 3.0)) / (1.0 + p_pade['b1'] * (log_lambda - 3.0))
    
    ax1.plot(lambda_range, delta_logistic * 100, 'b-', lw=2, label='Logistic (original)')
    ax1.plot(lambda_range, delta_poly * 100, 'r--', lw=2, label='Polynomial')
    ax1.plot(lambda_range, delta_pade * 100, 'g-.', lw=2, label='Padé [1/1]')
    
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('$\\tilde{\\Lambda}$', fontsize=12)
    ax1.set_ylabel('EOS Correction [%]', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title('Alternative EOS Correction Forms', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Residuals comparison
    models = ['M2_original', 'polynomial', 'pade', 'mass_dependent']
    chi2_values = [results[m]['chi2_dof'] for m in models]
    
    ax2.bar(range(len(models)), chi2_values, tick_label=models)
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='χ²/dof = 1')
    ax2.set_ylabel('χ²/dof', fontsize=12)
    ax2.set_title('Model Fit Quality', fontsize=14)
    ax2.legend()
    
    # Rotate labels
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_alternative_eos_forms.png'), dpi=300)
    plt.close()

def main():
    """Main workflow"""
    # Load data
    data_file = "./data/nr_simulations_with_f2.csv"
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Quality filtering
    if 'f2_err' in df.columns:
        error_threshold = df['f2_err'].quantile(0.95)
        df = df[df['f2_err'] < error_threshold]
        df = df[df['f2_err'] > 0]
        print(f"Filtered to {len(df)} simulations with reliable error estimates")
    
    # Test models
    results = test_models(df)
    
    # Generate comparison plots
    plot_alternative_eos_forms(df, results)
    
    print("\nAlternative model analysis complete!")

if __name__ == "__main__":
    main()