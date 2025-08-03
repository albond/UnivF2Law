#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Calibration script for universal f2 law with model comparison
Compares nested models using AIC/BIC criteria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os
import emcee
import time
from datetime import timedelta
from config import (ERROR_SCALING_FACTOR, INITIAL_PARAMS_M2_FIXED_W,
                    BOUNDS_M2_FIXED_W, calculate_aic_bic_likelihood, predict_f2, USE_ZETA_CORRECTION)

# Physical constants
G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
c = 299_792_458  # Speed of light [m/s]
M_sun = 1.98847e30  # Solar mass [kg]

# --- helper ----------------------------------------------------------------
def dhms(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    d  = td.days
    h, r = divmod(td.seconds, 3600)
    m, s = divmod(r, 60)
    out = []
    if d: out.append(f"{d} d")
    if d or h: out.append(f"{h:02d} h")
    out.append(f"{m:02d} m")
    out.append(f"{s:02d} s")
    return " ".join(out)

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

def compute_m_threshold(tilde_lambda):
    """
    Compute threshold mass for collapse to BH
    
    Parameters:
    -----------
    tilde_lambda : float
        Combined dimensionless tidal deformability
        
    Returns:
    --------
    m_threshold : float
        Threshold mass in solar masses [M_sun]
    """
    return 2.38 + 3.606e-4 * tilde_lambda**0.858

# Model definitions
def f_model_M0(x, alpha, beta):
    """M0: Base model with mass asymmetry only"""
    f_base, q, tilde_lambda = x
    return alpha * f_base * (1.0 + beta * (q - 1)**2)

def f_model_M1(x, alpha, beta, gamma):
    """M1: M0 + collapse correction"""
    f_base, q, tilde_lambda, m_tot = x[:4]
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # Collapse correction
    m_thresh = compute_m_threshold(tilde_lambda)
    delta_coll = np.where(m_tot > m_thresh, gamma * (m_tot / m_thresh - 1.0), 0.0)
    
    return f_model * (1.0 + delta_coll)

def f_model_M2(x, alpha, beta, delta, lambda0, w):
    """M2: M0 + EOS correction (optimal model)"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # EOS correction
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    
    return f_model * (1.0 + delta_eos)

def f_model_M2_fixed_w(x, alpha, beta, delta, lambda0):
    """M2 with fixed w=0.1 (4 parameters)"""
    f_base, q, tilde_lambda = x
    return predict_f2(f_base, q, tilde_lambda, alpha, beta, delta, lambda0, w=0.1)

def f_model_M2_fixed_w_zeta(x, alpha, beta, delta, lambda0, zeta):
    """M2 with fixed w=0.1 and systematic correction (5 parameters)"""
    f_base, q, tilde_lambda, code_indicator = x
    return predict_f2(f_base, q, tilde_lambda, alpha, beta, delta, lambda0, w=0.1, 
                     zeta=zeta, code_indicator=code_indicator)

def f_model_M3(x, alpha, beta, kappa, q0):
    """M3: M0 + asymmetry correction"""
    f_base, q, tilde_lambda = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # Asymmetry correction
    delta_asymm = np.where(q < q0, kappa * (q0 - q)**2, 0.0)
    
    return f_model * (1.0 + delta_asymm)

def f_model_full(x, alpha, beta, gamma, delta, lambda0, w, kappa, q0):
    """Full model with all corrections"""
    f_base, q, tilde_lambda, m_tot = x
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # Collapse correction
    m_thresh = compute_m_threshold(tilde_lambda)
    delta_coll = np.where(m_tot > m_thresh, gamma * (m_tot / m_thresh - 1.0), 0.0)
    
    # EOS correction
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    
    # Asymmetry correction
    delta_asymm = np.where(q < q0, kappa * (q0 - q)**2, 0.0)
    
    return f_model * (1.0 + delta_coll + delta_eos + delta_asymm)


def run_mcmc(f_obs, f_err, x_data, model_func, p0, param_names, bounds):
    """Run MCMC analysis for parameter uncertainties with convergence control"""
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
        # Bounds are already passed as list of tuples from calibrate_all_models
        for i, val in enumerate(theta):
            if not (bounds[i][0] <= val <= bounds[i][1]):
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
            # Check bounds (bounds are already list of tuples)
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
    
    # Initial burn-in phase with adaptive step size
    print("Running burn-in phase with adaptive steps...")
    
    # First, find reasonable step size
    test_steps = 100
    pos, prob, state = sampler.run_mcmc(pos, test_steps, progress=False)
    initial_acc = np.mean(sampler.acceptance_fraction)
    print(f"Initial acceptance fraction: {initial_acc:.3f}")
    
    # Adjust walker initialization if acceptance is too low
    if initial_acc < 0.1:
        print("Low acceptance detected, reinitializing walkers closer to optimum...")
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
                
                # bounds are already list of tuples
                if all(bounds[j][0] <= trial[j] <= bounds[j][1] for j in range(ndim)):
                    pos.append(trial)
                    break
        pos = np.array(pos)
    
    # Continue burn-in
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, 1000, progress=True)
    sampler.reset()
    
    # Production run with convergence check
    print("Running production phase...")
    max_steps = 2000  # Default production steps
    check_interval = 500
    autocorr_tol = 50.0
    
    for i in range(0, max_steps, check_interval):
        pos, prob, state = sampler.run_mcmc(pos, check_interval, progress=True)
        
        # Check convergence using autocorrelation time
        try:
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * autocorr_tol < sampler.iteration)
            
            print(f"\nStep {sampler.iteration}:")
            print(f"  Autocorrelation times: {tau}")
            print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
            
            if converged:
                print("  Chains have converged!")
                break
        except emcee.autocorr.AutocorrError:
            print(f"\nStep {sampler.iteration}: Not enough samples for autocorrelation estimate")
            continue
    
    # Check final convergence diagnostics
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
    print(f"\nFinal acceptance fraction: {acceptance_fraction:.3f}")
    
    if acceptance_fraction < 0.2:
        print("Warning: Low acceptance fraction. Consider adjusting proposal scale.")
    elif acceptance_fraction > 0.5:
        print("Warning: High acceptance fraction. Consider adjusting proposal scale.")
    
    # Get samples with appropriate thinning
    try:
        tau = sampler.get_autocorr_time()
        thin = int(np.max(tau) / 2)
        discard = int(2 * np.max(tau))
        print(f"\nUsing discard={discard}, thin={thin} based on autocorrelation analysis")
    except:
        thin = 15
        discard = 1000
        print(f"\nUsing default discard={discard}, thin={thin}")
    
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    # Calculate Gelman-Rubin statistic for convergence check
    chains = sampler.get_chain(discard=discard)[:, :, :]  # shape: (steps, walkers, params)
    n_steps, n_walkers, _ = chains.shape
    
    # Split chains for Gelman-Rubin
    mid = n_walkers // 2
    chain1 = chains[:, :mid, :].reshape(-1, ndim)
    chain2 = chains[:, mid:, :].reshape(-1, ndim)
    
    # Calculate R-hat (Gelman-Rubin statistic)
    mean1 = np.mean(chain1, axis=0)
    mean2 = np.mean(chain2, axis=0)
    var1 = np.var(chain1, axis=0)
    var2 = np.var(chain2, axis=0)
    
    mean_all = np.mean(samples, axis=0)
    B = n_steps * ((mean1 - mean_all)**2 + (mean2 - mean_all)**2)
    W = 0.5 * (var1 + var2)
    V = (1 - 1/n_steps) * W + B/n_steps
    R_hat = np.sqrt(V / W)
    
    print(f"\nGelman-Rubin R-hat values: {R_hat}")
    if np.any(R_hat > 1.1):
        print("Warning: Some R-hat values > 1.1, indicating incomplete convergence")
    
    # Calculate statistics
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    medians = percentiles[1]
    uncertainties = np.array([medians - percentiles[0], percentiles[2] - medians])
    
    # Save chain diagnostics plot
    fig, axes = plt.subplots(ndim, 2, figsize=(12, 3*ndim))
    
    # Plot chains and autocorrelation
    for i in range(ndim):
        # Walker traces
        axes[i, 0].plot(sampler.get_chain()[:, :, i], alpha=0.3, color='black', linewidth=0.5)
        axes[i, 0].set_ylabel(param_names[i])
        axes[i, 0].set_xlabel('Step')
        axes[i, 0].axvline(discard, color='red', linestyle='--', label='Burn-in cutoff')
        
        # Autocorrelation function
        from emcee import autocorr
        chain_i = sampler.get_chain()[:, :, i].flatten()
        acf = autocorr.function_1d(chain_i)[:100]
        axes[i, 1].plot(acf)
        axes[i, 1].set_ylabel('ACF')
        axes[i, 1].set_xlabel('Lag')
        axes[i, 1].axhline(0, color='black', linestyle='--')
    
    axes[0, 0].set_title('Walker Traces')
    axes[0, 1].set_title('Autocorrelation Functions')
    plt.tight_layout()
    plt.savefig('./figs/mcmc_diagnostics.png', dpi=150)
    plt.close()
    
    return medians, uncertainties, samples

def calibrate_all_models(df):
    """Calibrate all nested models and compute information criteria"""
    # Prepare data
    # Extract masses ensuring m1 >= m2 (in solar masses)
    m1 = df[['m1', 'm2']].max(axis=1).values  # Primary mass [M_sun]
    m2 = df[['m1', 'm2']].min(axis=1).values  # Secondary mass [M_sun]
    lambda1 = np.where(df['m1'] >= df['m2'], df['lambda1'], df['lambda2']).astype(float)
    lambda2 = np.where(df['m1'] >= df['m2'], df['lambda2'], df['lambda1']).astype(float)
    
    q = m2 / m1  # Mass ratio (q <= 1)
    M_tot = m1 + m2  # Total mass [M_sun]
    tilde_Lambda = compute_tilde_lambda(m1, m2, lambda1, lambda2)  # Dimensionless
    
    # Calculate effective radius and base frequency
    # R_L: Effective radius based on tidal deformability [meters]
    R_L = tilde_Lambda**(1/5) * (G * (M_tot * M_sun)) / c**2
    # f_base: Keplerian frequency at R_L [Hz]
    f_base = np.sqrt(G * (M_tot * M_sun) / R_L**3) / (2 * np.pi)
    
    f_obs = df['f2_NR'].values
    # Scale errors to achieve χ²/dof ≈ 1 (scaling factor from error audit)
    f_err_raw = df.get('f2_err', np.full(len(df), 50.0)).values
    error_scaling = ERROR_SCALING_FACTOR  # From config.py: 0.997
    f_err = f_err_raw * error_scaling
    
    n_data = len(f_obs)
    results = {}
    
    print("\nCalibrating nested models...")
    
    # M0: Base model (2 params)
    print("\n1. M0: Base model (α, β)")
    x_M0 = (f_base, q, tilde_Lambda)
    p0_M0 = [4.5, -0.8]
    bounds_M0 = ([0.1, -20], [10, 20])  # Extended beta bounds
    
    bounds_M0_tuple = (bounds_M0[0], bounds_M0[1])  # Convert to tuple format
    popt_M0, pcov_M0 = curve_fit(f_model_M0, x_M0, f_obs, p0=p0_M0, 
                                  sigma=f_err, absolute_sigma=True, bounds=bounds_M0_tuple)
    
    f_pred_M0 = f_model_M0(x_M0, *popt_M0)
    chi2_M0 = np.sum(((f_obs - f_pred_M0) / f_err)**2)
    aic_M0, bic_M0, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M0, f_err, 2)
    
    results['M0'] = {
        'params': dict(zip(['alpha', 'beta'], popt_M0.tolist())),
        'chi2': float(chi2_M0),
        'chi2_dof': float(chi2_M0 / (n_data - 2)),
        'aic': float(aic_M0),
        'bic': float(bic_M0)
    }
    
    # M1: + collapse correction (3 params)
    print("\n2. M1: M0 + collapse correction (α, β, γ)")
    x_M1 = (f_base, q, tilde_Lambda, M_tot)
    p0_M1 = [4.5, -0.8, 0.1]
    bounds_M1 = ([0.1, -20, -1], [10, 20, 1])  # Extended beta bounds
    
    bounds_M1_tuple = (bounds_M1[0], bounds_M1[1])  # Convert to tuple format
    popt_M1, pcov_M1 = curve_fit(f_model_M1, x_M1, f_obs, p0=p0_M1,
                                  sigma=f_err, absolute_sigma=True, bounds=bounds_M1_tuple)
    
    f_pred_M1 = f_model_M1(x_M1, *popt_M1)
    chi2_M1 = np.sum(((f_obs - f_pred_M1) / f_err)**2)
    aic_M1, bic_M1, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M1, f_err, 3)
    
    results['M1'] = {
        'params': dict(zip(['alpha', 'beta', 'gamma'], popt_M1.tolist())),
        'chi2': float(chi2_M1),
        'chi2_dof': float(chi2_M1 / (n_data - 3)),
        'aic': float(aic_M1),
        'bic': float(bic_M1)
    }
    
    # M2: + EOS correction (5 params) - This will be the optimal model
    print("\n3. M2: M0 + EOS correction (α, β, δ, λ₀, w)")
    x_M2 = (f_base, q, tilde_Lambda)
    p0_M2 = [4.5, -0.8, -0.5, 3.1, 0.1]
    bounds_M2 = ([0.1, -20, -1, 2, 0.05], [10, 20, 1, 3.5, 1])  # Extended beta bounds
    
    bounds_M2_tuple = (bounds_M2[0], bounds_M2[1])  # Convert to tuple format
    popt_M2, pcov_M2 = curve_fit(f_model_M2, x_M2, f_obs, p0=p0_M2,
                                  sigma=f_err, absolute_sigma=True, bounds=bounds_M2_tuple)
    
    f_pred_M2 = f_model_M2(x_M2, *popt_M2)
    chi2_M2 = np.sum(((f_obs - f_pred_M2) / f_err)**2)
    aic_M2, bic_M2, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M2, f_err, 5)
    
    results['M2'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'w'], popt_M2.tolist())),
        'chi2': float(chi2_M2),
        'chi2_dof': float(chi2_M2 / (n_data - 5)),
        'aic': float(aic_M2),
        'bic': float(bic_M2)
    }
    
    # M3: + asymmetry correction (4 params)
    print("\n4. M3: M0 + asymmetry correction (α, β, κ, q₀)")
    x_M3 = (f_base, q, tilde_Lambda)
    p0_M3 = [4.5, -0.8, 0.5, 0.85]
    bounds_M3 = ([0.1, -20, -5, 0.7], [10, 20, 5, 1.0])  # Extended beta bounds
    
    bounds_M3_tuple = (bounds_M3[0], bounds_M3[1])  # Convert to tuple format
    popt_M3, pcov_M3 = curve_fit(f_model_M3, x_M3, f_obs, p0=p0_M3,
                                  sigma=f_err, absolute_sigma=True, bounds=bounds_M3_tuple)
    
    f_pred_M3 = f_model_M3(x_M3, *popt_M3)
    chi2_M3 = np.sum(((f_obs - f_pred_M3) / f_err)**2)
    aic_M3, bic_M3, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M3, f_err, 4)
    
    results['M3'] = {
        'params': dict(zip(['alpha', 'beta', 'kappa', 'q0'], popt_M3.tolist())),
        'chi2': float(chi2_M3),
        'chi2_dof': float(chi2_M3 / (n_data - 4)),
        'aic': float(aic_M3),
        'bic': float(bic_M3)
    }
    
    # M_full: All corrections (8 params)
    print("\n5. M_full: All corrections (α, β, γ, δ, λ₀, w, κ, q₀)")
    x_full = (f_base, q, tilde_Lambda, M_tot)
    p0_full = [4.5, -0.8, 0.1, -0.5, 3.1, 0.1, 0.5, 0.85]
    bounds_full = ([0.1, -20, -1, -1, 2, 0.05, -5, 0.7], 
                   [10, 20, 1, 1, 3.5, 1, 5, 1.0])  # Extended beta bounds
    
    bounds_full_tuple = (bounds_full[0], bounds_full[1])  # Convert to tuple format
    popt_full, pcov_full = curve_fit(f_model_full, x_full, f_obs, p0=p0_full,
                                      sigma=f_err, absolute_sigma=True, bounds=bounds_full_tuple)
    
    f_pred_full = f_model_full(x_full, *popt_full)
    chi2_full = np.sum(((f_obs - f_pred_full) / f_err)**2)
    aic_full, bic_full, _ = calculate_aic_bic_likelihood(f_obs, f_pred_full, f_err, 8)
    
    results['M_full'] = {
        'params': dict(zip(['alpha', 'beta', 'gamma', 'delta', 'lambda0', 'w', 'kappa', 'q0'], 
                          popt_full.tolist())),
        'chi2': float(chi2_full),
        'chi2_dof': float(chi2_full / (n_data - 8)),
        'aic': float(aic_full),
        'bic': float(bic_full)
    }
    
    # Test M2 with fixed w=0.1 (4 parameters)
    print("\n3b. M2_fixed_w: M0 + EOS correction with w=0.1 (α, β, δ, λ₀)")
    p0_M2_fixed = INITIAL_PARAMS_M2_FIXED_W  # Use reference parameters from config
    bounds_M2_fixed = ([b[0] for b in zip(*BOUNDS_M2_FIXED_W)], [b[1] for b in zip(*BOUNDS_M2_FIXED_W)])
    
    bounds_M2_fixed_tuple = (bounds_M2_fixed[0], bounds_M2_fixed[1])  # Convert to tuple format
    popt_M2_fixed, pcov_M2_fixed = curve_fit(f_model_M2_fixed_w, x_M2, f_obs, p0=p0_M2_fixed,
                                              sigma=f_err, absolute_sigma=True, bounds=bounds_M2_fixed_tuple)
    
    f_pred_M2_fixed = f_model_M2_fixed_w(x_M2, *popt_M2_fixed)
    chi2_M2_fixed = np.sum(((f_obs - f_pred_M2_fixed) / f_err)**2)
    aic_M2_fixed, bic_M2_fixed, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M2_fixed, f_err, 4)
    
    results['M2_fixed_w'] = {
        'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0'], popt_M2_fixed.tolist())),
        'chi2': float(chi2_M2_fixed),
        'chi2_dof': float(chi2_M2_fixed / (n_data - 4)),
        'aic': float(aic_M2_fixed),
        'bic': float(bic_M2_fixed)
    }
    
    # Conditionally test M2_fixed_w with zeta correction
    if USE_ZETA_CORRECTION:
        print("\n3c. M2_fixed_w_zeta: M2_fixed_w + systematic correction (α, β, δ, λ₀, ζ)")
        
        # Create code indicator: +1 for THC, -1 for BAM
        code_indicator = np.where(df['model_name'].str.contains('THC', case=False), 1.0, -1.0)
        x_M2_zeta = (f_base, q, tilde_Lambda, code_indicator)
        
        # Start from M2_fixed_w parameters plus small zeta
        p0_M2_zeta = list(popt_M2_fixed) + [0.0]
        bounds_lower = bounds_M2_fixed[0] + [-0.5]
        bounds_upper = bounds_M2_fixed[1] + [0.5]
        bounds_M2_zeta_tuple = (bounds_lower, bounds_upper)
        
        popt_M2_zeta, pcov_M2_zeta = curve_fit(f_model_M2_fixed_w_zeta, x_M2_zeta, f_obs, 
                                                p0=p0_M2_zeta, sigma=f_err, absolute_sigma=True, 
                                                bounds=bounds_M2_zeta_tuple)
        
        f_pred_M2_zeta = f_model_M2_fixed_w_zeta(x_M2_zeta, *popt_M2_zeta)
        chi2_M2_zeta = np.sum(((f_obs - f_pred_M2_zeta) / f_err)**2)
        aic_M2_zeta, bic_M2_zeta, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M2_zeta, f_err, 5)
        
        results['M2_fixed_w_zeta'] = {
            'params': dict(zip(['alpha', 'beta', 'delta', 'lambda0', 'zeta'], popt_M2_zeta.tolist())),
            'chi2': float(chi2_M2_zeta),
            'chi2_dof': float(chi2_M2_zeta / (n_data - 5)),
            'aic': float(aic_M2_zeta),
            'bic': float(bic_M2_zeta)
        }
        
        # Print zeta analysis results
        print(f"\nSystematic correction analysis:")
        print(f"  ζ = {popt_M2_zeta[4]:.4f}")
        print(f"  χ²/dof without ζ: {chi2_M2_fixed/(n_data-4):.3f}")
        print(f"  χ²/dof with ζ: {chi2_M2_zeta/(n_data-5):.3f}")
        print(f"  Δχ²/dof = {chi2_M2_zeta/(n_data-5) - chi2_M2_fixed/(n_data-4):.4f}")
        print(f"  ΔAIC = {aic_M2_zeta - aic_M2_fixed:.1f}")
        print(f"  ΔBIC = {bic_M2_zeta - bic_M2_fixed:.1f}")
    
    # Find optimal model
    min_aic = min(results[m]['aic'] for m in results)
    min_bic = min(results[m]['bic'] for m in results)
    
    # Add delta values
    for model in results:
        results[model]['delta_aic'] = results[model]['aic'] - min_aic
        results[model]['delta_bic'] = results[model]['bic'] - min_bic
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'k':<3} {'χ²/dof':<8} {'AIC':<10} {'ΔAIC':<10} {'BIC':<10} {'ΔBIC':<10}")
    print("-"*80)
    
    model_list = ['M0', 'M1', 'M2', 'M2_fixed_w', 'M3', 'M_full']
    if USE_ZETA_CORRECTION and 'M2_fixed_w_zeta' in results:
        model_list.insert(4, 'M2_fixed_w_zeta')  # Insert after M2_fixed_w
    
    for model in model_list:
        r = results[model]
        n_params = len(r['params'])
        print(f"{model:<15} {n_params:<3} {r['chi2_dof']:<8.2f} {r['aic']:<10.1f} "
              f"{r['delta_aic']:<10.1f} {r['bic']:<10.1f} {r['delta_bic']:<10.1f}")
    
    # Identify optimal model based on BIC (penalizes complexity)
    optimal_by_bic = min(results.items(), key=lambda x: x[1]['bic'])[0]
    print(f"\nOptimal model by BIC: {optimal_by_bic}")
    
    # Use the model selected by BIC
    optimal_model = optimal_by_bic
    
    # Run benchmark for the selected model
    if optimal_model == 'M2':
        # Run MCMC benchmark without "first step is slow" to get ETA for normal run
        warmup_steps = 2  # warming up, not taken into account in time
        bench_steps = 10  # real steps to benchmark
        burn_in_steps = 1000  # planned burn-in
        production_steps = 10000  # planned production
        total_steps = burn_in_steps + production_steps

        print(f"\n⏳ Benchmarking {bench_steps} steps (after {warmup_steps} warm-up)…")
        
        # Setup minimal MCMC for benchmarking
        ndim = 5  # M2 has 5 parameters
        nwalkers = 2 * ndim * (ndim + 1)  # 60 walkers
        
        # Simple log probability for benchmark (just chi2)
        def log_prob_bench(theta):
            bounds_list = list(zip(bounds_M2[0], bounds_M2[1]))
            if not all(bounds_list[i][0] <= theta[i] <= bounds_list[i][1] for i in range(ndim)):
                return -np.inf
            f_pred = f_model_M2(x_M2, *theta)
            chi2 = np.sum(((f_obs - f_pred) / f_err)**2)
            return -0.5 * chi2
        
        # Initialize walkers with proper scatter
        pos_bench = []
        for i in range(nwalkers):
            trial = popt_M2.copy()
            for j in range(ndim):
                scale = 0.01 * max(abs(popt_M2[j]), 0.1)
                trial[j] += scale * np.random.randn()
            pos_bench.append(trial)
        pos_bench = np.array(pos_bench)
        
        # Create sampler with same moves as main MCMC
        moves = [
            (emcee.moves.StretchMove(a=2.0), 0.8),
            (emcee.moves.StretchMove(a=1.5), 0.2),
        ]
        sampler_bench = emcee.EnsembleSampler(nwalkers, ndim, log_prob_bench, moves=moves)
        
        # 1) Warm-up (not timed)
        print(f"  Warming up with {warmup_steps} steps...")
        pos_bench, _, _ = sampler_bench.run_mcmc(pos_bench, warmup_steps, progress=False)
        
        # 2) Benchmark
        print(f"  Running {bench_steps} benchmark steps...")
        t0 = time.perf_counter()
        pos_bench, _, _ = sampler_bench.run_mcmc(pos_bench, bench_steps, progress=False)
        dt_total = time.perf_counter() - t0
        
        # 3) Calculate ETA
        dt_avg = dt_total / bench_steps
        # Total steps includes:
        # - test_steps (100) for initial acceptance check
        # - burn_in_steps (1000) for actual burn-in
        # - production_steps with convergence checks
        # Note: actual time will be longer due to progress bar overhead and convergence checks
        total_mcmc_steps = 100 + burn_in_steps + production_steps  # test + burn-in + production
        eta_sec = dt_avg * total_mcmc_steps * 1.2  # Add 20% overhead for progress bars and checks
        
        print(f"\n📊 Benchmark results:")
        print(f"  Average time per step: {dt_avg:.3f} seconds")
        print(f"  Total MCMC steps: ~{total_mcmc_steps} (test: 100, burn-in: {burn_in_steps}, production: {production_steps})")
        print(f"  Estimated total time: {dhms(eta_sec)} (including overhead)")
        print(f"  Acceptance fraction: {np.mean(sampler_bench.acceptance_fraction):.3f}")
        
        # Warn if time is too long
        if eta_sec > 3600:  # More than 1 hour
            print(f"\n⚠️  WARNING: MCMC will take a long time! Consider:")
            print(f"  - Reducing number of steps")
            print(f"  - Using fewer walkers")
            print(f"  - Running on a faster machine")
            if eta_sec > 86400:  # More than 1 day
                print(f"\n🛑 ETA exceeds 1 day. You may want to reconsider the analysis strategy.")
        
        print(f"\n" + "="*60)

    else:
        optimal_model = optimal_by_bic
        
        # Run benchmark for the optimal model
        if optimal_model == 'M2_fixed_w':
            warmup_steps = 2
            bench_steps = 10
            burn_in_steps = 1000
            production_steps = 2000  # Reduced for faster results
            total_steps = burn_in_steps + production_steps
            
            print(f"\n⏳ Benchmarking {bench_steps} steps for M2_fixed_w (after {warmup_steps} warm-up)…")
            
            ndim = 4  # M2_fixed_w has 4 parameters
            nwalkers = 2 * ndim * (ndim + 1)
            
            def log_prob_bench(theta):
                bounds_list = list(zip(bounds_M2_fixed[0], bounds_M2_fixed[1]))
                if not all(bounds_list[i][0] <= theta[i] <= bounds_list[i][1] for i in range(ndim)):
                    return -np.inf
                try:
                    f_pred = f_model_M2_fixed_w(x_M2, *theta)
                    chi2 = np.sum(((f_obs - f_pred) / f_err)**2)
                    return -0.5 * chi2
                except:
                    return -np.inf
            
            # Initialize walkers
            pos_bench = []
            for i in range(nwalkers):
                trial = popt_M2_fixed.copy()
                for j in range(ndim):
                    scale = 0.01 * max(abs(popt_M2_fixed[j]), 0.1)
                    trial[j] += scale * np.random.randn()
                pos_bench.append(trial)
            pos_bench = np.array(pos_bench)
            
            # Run benchmark
            sampler_bench = emcee.EnsembleSampler(nwalkers, ndim, log_prob_bench)
            
            print(f"  Warming up with {warmup_steps} steps...")
            pos_bench, _, _ = sampler_bench.run_mcmc(pos_bench, warmup_steps, progress=False)
            
            print(f"  Running {bench_steps} benchmark steps...")
            t0 = time.perf_counter()
            pos_bench, _, _ = sampler_bench.run_mcmc(pos_bench, bench_steps, progress=False)
            dt_total = time.perf_counter() - t0
            
            dt_avg = dt_total / bench_steps
            # Total steps includes test + burn-in + production with overhead
            total_mcmc_steps = 100 + burn_in_steps + production_steps
            eta_sec = dt_avg * total_mcmc_steps * 1.2  # Add 20% overhead
            
            print(f"\n📊 Benchmark results:")
            print(f"  Average time per step: {dt_avg:.3f} seconds")
            print(f"  Total MCMC steps: ~{total_mcmc_steps} (test: 100, burn-in: {burn_in_steps}, production: {production_steps})")
            print(f"  Estimated total time: {dhms(eta_sec)} (including overhead)")
            print(f"  Acceptance fraction: {np.mean(sampler_bench.acceptance_fraction):.3f}")
            
            print(f"\n" + "="*60)
    
    print(f"Selected model for detailed analysis: {optimal_model}")
    
    # Run MCMC for optimal model
    print(f"\nRunning MCMC for optimal model {optimal_model}...")
    
    if optimal_model == 'M2':
        param_names = ['alpha', 'beta', 'delta', 'lambda0', 'w']
        bounds_M2_list = list(zip(bounds_M2[0], bounds_M2[1]))  # Convert for MCMC
        medians, uncertainties, samples = run_mcmc(
            f_obs, f_err, x_M2, f_model_M2, popt_M2, param_names, bounds_M2_list
        )
        f_pred_optimal = f_model_M2(x_M2, *medians)
    elif optimal_model == 'M2_fixed_w':
        param_names = ['alpha', 'beta', 'delta', 'lambda0']
        bounds_M2_fixed_list = list(zip(bounds_M2_fixed[0], bounds_M2_fixed[1]))  # Convert for MCMC
        # Use reference parameters from config as starting point for MCMC
        medians, uncertainties, samples = run_mcmc(
            f_obs, f_err, x_M2, f_model_M2_fixed_w, popt_M2_fixed, param_names, bounds_M2_fixed_list
        )
        f_pred_optimal = f_model_M2_fixed_w(x_M2, *medians)
        
        # Ensure consistency: update model_comparison.json with consistent parameters
        results['M2_fixed_w']['params'] = dict(zip(param_names, medians.tolist()))
        results['M2_fixed_w']['chi2'] = float(np.sum(((f_obs - f_pred_optimal) / f_err)**2))
        results['M2_fixed_w']['chi2_dof'] = float(results['M2_fixed_w']['chi2'] / (n_data - 4))
        aic_updated, bic_updated, _ = calculate_aic_bic_likelihood(f_obs, f_pred_M2_fixed, f_err, 4)
        results['M2_fixed_w']['aic'] = float(aic_updated)
        results['M2_fixed_w']['bic'] = float(bic_updated)
    elif optimal_model == 'M2_fixed_w_zeta':
        param_names = ['alpha', 'beta', 'delta', 'lambda0', 'zeta']
        bounds_M2_zeta_list = list(zip(bounds_lower, bounds_upper))  # Convert for MCMC
        medians, uncertainties, samples = run_mcmc(
            f_obs, f_err, x_M2_zeta, f_model_M2_fixed_w_zeta, popt_M2_zeta, param_names, bounds_M2_zeta_list
        )
        f_pred_optimal = f_model_M2_fixed_w_zeta(x_M2_zeta, *medians)
        
        # Update results
        results['M2_fixed_w_zeta']['params'] = dict(zip(param_names, medians.tolist()))
        results['M2_fixed_w_zeta']['chi2'] = float(np.sum(((f_obs - f_pred_optimal) / f_err)**2))
        results['M2_fixed_w_zeta']['chi2_dof'] = float(results['M2_fixed_w_zeta']['chi2'] / (n_data - 5))
        aic_updated, bic_updated, _ = calculate_aic_bic_likelihood(f_obs, f_pred_optimal, f_err, 5)
        results['M2_fixed_w_zeta']['aic'] = float(aic_updated)
        results['M2_fixed_w_zeta']['bic'] = float(bic_updated)
    else:
        # Fallback to M2
        param_names = ['alpha', 'beta', 'delta', 'lambda0', 'w']
        bounds_M2_list = list(zip(bounds_M2[0], bounds_M2[1]))  # Convert for MCMC
        medians, uncertainties, samples = run_mcmc(
            f_obs, f_err, x_M2, f_model_M2, popt_M2, param_names, bounds_M2_list
        )
        f_pred_optimal = f_model_M2(x_M2, *medians)
    
    # Calculate statistics for optimal model
    residuals = f_pred_optimal - f_obs
    rel_residuals = residuals / f_obs
    rms_error = np.sqrt(np.mean(residuals**2))
    mean_rel_error = np.mean(np.abs(rel_residuals))
    
    # Save results
    # Store chi2/dof information
    chi2_optimal = results[optimal_model]['chi2']
    dof_optimal = n_data - len(results[optimal_model]['params'])
    chi2_dof_optimal = chi2_optimal / dof_optimal
    
    optimal_result = {
        'model_type': optimal_model,
        'description': f'Optimal model ({optimal_model}) selected by AIC/BIC criteria',
        'parameters': dict(zip(param_names, medians.tolist())),
        'uncertainties': dict(zip([f"{p}_err" for p in param_names], 
                                 uncertainties[0].tolist())),
        'bayesian_estimates': {
            param: np.percentile(samples[:, i], [16, 50, 84]).tolist()
            for i, param in enumerate(param_names)
        },
        'statistics': {
            'chi2': float(chi2_optimal),
            'dof': dof_optimal,
            'chi2_per_dof': float(chi2_dof_optimal),
            'rms_error_Hz': float(rms_error),
            'mean_relative_error': float(mean_rel_error)
        },
        'physical_interpretation': {
            'eos_transition_lambda': float(10**medians[3]) if optimal_model == 'M2' else float(10**medians[3]),
            'transition_width': float(medians[4]) if optimal_model == 'M2' else 0.1,  # Fixed for M2_fixed_w
            'max_eos_effect_percent': float(abs(medians[2]) * 100)
        }
    }
    
    # Save both comparison and optimal model results
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, 'optimal_model.json'), 'w') as f:
        json.dump(optimal_result, f, indent=2)
    
    # Save MCMC samples for later use
    np.save(os.path.join(output_dir, 'mcmc_samples_M2.npy'), samples)
    print(f"\nMCMC samples saved to mcmc_samples_M2.npy")
    
    print(f"\nResults saved to {output_dir}")
    
    return results, optimal_result

def main():
    """Main calibration workflow"""
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
    
    # Run calibration
    results, optimal = calibrate_all_models(df)
    
    print("\nCalibration complete!")

if __name__ == "__main__":
    main()