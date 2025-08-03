#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate figures for the optimal model
Focus on M2 model selected by AIC/BIC criteria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import json
import os
from sklearn.model_selection import KFold
from config import M2_FIXED_W_PARAMS, ERROR_SCALING_FACTOR, INITIAL_PARAMS_M2_FIXED_W, predict_f2

# Optional imports with error handling
try:
    import corner
    HAS_CORNER = True
except ImportError:
    print("Warning: corner package not installed. Some plots will be skipped.")
    print("Install with: pip install corner")
    HAS_CORNER = False

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    print("Warning: emcee package not installed. MCMC diagnostics will be limited.")
    print("Install with: pip install emcee")
    HAS_EMCEE = False

# Set plot style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

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

def f_model_optimal(x, alpha, beta, delta, lambda0, w=0.1):
    """Optimal model (M2 or M2_fixed_w)"""
    f_base, q, tilde_lambda = x
    return predict_f2(f_base, q, tilde_lambda, alpha, beta, delta, lambda0, w)

def load_data_and_params():
    """Load simulation data and optimal model parameters"""
    data_file = "./data/nr_simulations_with_f2.csv"
    params_file = "./results/optimal_model.json"
    
    df = pd.read_csv(data_file)
    
    # Quality filtering
    if 'f2_err' in df.columns:
        error_threshold = df['f2_err'].quantile(0.95)
        df = df[df['f2_err'] < error_threshold]
        df = df[df['f2_err'] > 0]
    
    # Load parameters but override with reference values to ensure consistency
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Override with reference parameters if M2_fixed_w
    if params['model_type'] == 'M2_fixed_w':
        params['parameters'] = M2_FIXED_W_PARAMS.copy()
    
    return df, params

def prepare_data(df):
    """Prepare derived quantities from raw data"""
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
    
    return {
        'm1': m1, 'm2': m2, 'q': q, 'M_tot': M_tot,
        'lambda1': lambda1, 'lambda2': lambda2,
        'tilde_Lambda': tilde_Lambda, 'f_base': f_base,
        'f_obs': df['f2_NR'].values,
        'f_err': df.get('f2_err', np.full(len(df), 50.0)).values * ERROR_SCALING_FACTOR
    }

def fig_1_calibration_scatter(data, params, output_dir):
    """Figure 1: Calibration quality scatter plot for optimal model"""
    p = params['parameters']
    # Handle both M2 and M2_fixed_w models
    if 'w' in p:
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0'], p['w']
        )
    else:
        # M2_fixed_w uses fixed w=0.1
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0']
        )
    
    plt.figure(figsize=(8, 8))
    
    # Main scatter
    plt.scatter(data['f_obs'], f_pred, alpha=0.6, s=50)
    
    # Perfect correlation line
    min_f = min(data['f_obs'].min(), f_pred.min())
    max_f = max(data['f_obs'].max(), f_pred.max())
    plt.plot([min_f, max_f], [min_f, max_f], 'k--', lw=2, label='Perfect correlation')
    
    # RMS error band
    rms_error = params['statistics']['rms_error_Hz']
    plt.fill_between([min_f, max_f], 
                     [min_f - rms_error, max_f - rms_error],
                     [min_f + rms_error, max_f + rms_error],
                     alpha=0.2, color='gray', label=f'±RMS ({rms_error:.0f} Hz)')
    
    plt.xlabel('$f_{2,NR}$ [Hz]', fontsize=12)
    plt.ylabel('$f_{2,model}$ [Hz]', fontsize=12)
    plt.title('Model Calibration Quality', fontsize=14)
    plt.legend()
    
    # Add statistics
    mean_err = params['statistics']['mean_relative_error'] * 100
    chi2_dof = params['statistics']['chi2_per_dof']
    plt.text(0.05, 0.95, f'χ²/dof = {chi2_dof:.2f}\nMean error: {mean_err:.1f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_1_calibration_scatter.png'), dpi=300)
    plt.close()

def fig_2_residuals_hist(data, params, output_dir):
    """Figure 2: Residuals histogram for optimal model"""
    p = params['parameters']
    # Handle both M2 and M2_fixed_w models
    if 'w' in p:
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0'], p['w']
        )
    else:
        # M2_fixed_w uses fixed w=0.1
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0']
        )
    rel_residuals = (f_pred - data['f_obs']) / data['f_obs']
    
    plt.figure(figsize=(8, 6))
    
    # Histogram
    n, bins, patches = plt.hist(rel_residuals, bins=30, density=True, 
                                alpha=0.7, edgecolor='black')
    
    # Fit Gaussian
    mu, sigma = stats.norm.fit(rel_residuals)
    x = np.linspace(rel_residuals.min(), rel_residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
             label=f'Gaussian fit\nμ = {mu:.4f}\nσ = {sigma:.4f}')
    
    plt.xlabel('$\\Delta f_2 / f_2$', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Relative Residuals', fontsize=14)
    plt.legend()
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_2_residuals_hist.png'), dpi=300)
    plt.close()

def fig_3_residuals_systematics(data, params, output_dir):
    """Figures 3a-c: Check for systematic trends in residuals"""
    p = params['parameters']
    # Handle both M2 and M2_fixed_w models
    if 'w' in p:
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0'], p['w']
        )
    else:
        # M2_fixed_w uses fixed w=0.1
        f_pred = f_model_optimal(
            (data['f_base'], data['q'], data['tilde_Lambda']),
            p['alpha'], p['beta'], p['delta'], p['lambda0']
        )
    rel_residuals = (f_pred - data['f_obs']) / data['f_obs']
    
    # 3a: vs tilde Lambda
    plt.figure(figsize=(8, 6))
    plt.scatter(data['tilde_Lambda'], rel_residuals, alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Add rolling average to show any remaining trends
    sorted_idx = np.argsort(data['tilde_Lambda'])
    window = len(data['tilde_Lambda']) // 10
    rolling_mean = pd.Series(rel_residuals[sorted_idx]).rolling(window, center=True).mean()
    plt.plot(data['tilde_Lambda'][sorted_idx], rolling_mean, 'r-', lw=2, 
             label='Rolling average')
    
    plt.xlabel('$\\tilde{\\Lambda}$', fontsize=12)
    plt.ylabel('$\\Delta f_2 / f_2$', fontsize=12)
    plt.title('Residuals vs Tidal Deformability', fontsize=14)
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_3a_residuals_vs_Lambda.png'), dpi=300)
    plt.close()
    
    # 3b: vs q
    plt.figure(figsize=(8, 6))
    plt.scatter(data['q'], rel_residuals, alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('$q$', fontsize=12)
    plt.ylabel('$\\Delta f_2 / f_2$', fontsize=12)
    plt.title('Residuals vs Mass Ratio', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_3b_residuals_vs_q.png'), dpi=300)
    plt.close()
    
    # 3c: vs M_tot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['M_tot'], rel_residuals, alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('$M_{tot}$ [$M_\\odot$]', fontsize=12)
    plt.ylabel('$\\Delta f_2 / f_2$', fontsize=12)
    plt.title('Residuals vs Total Mass', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_3c_residuals_vs_Mtot.png'), dpi=300)
    plt.close()

def fig_4_eos_correction_analysis(data, params, output_dir):
    """Figure 4: EOS correction analysis"""
    p = params['parameters']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: EOS correction function
    lambda_range = np.logspace(1.5, 3.5, 1000)
    # Handle both M2 and M2_fixed_w models
    w_value = p.get('w', 0.1)  # Default to 0.1 if w not in parameters
    delta_eos = p['delta'] / (1.0 + np.exp((np.log10(lambda_range) - p['lambda0']) / w_value))
    
    ax1.plot(lambda_range, delta_eos * 100, 'b-', lw=2)
    ax1.axvline(10**p['lambda0'], color='r', linestyle='--', alpha=0.7,
                label=f'Transition at Λ̃={10**p["lambda0"]:.0f}')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('$\\tilde{\\Lambda}$', fontsize=12)
    ax1.set_ylabel('EOS Correction [%]', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title('EOS Correction Profile', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Data distribution colored by EOS correction magnitude
    eos_corr_data = p['delta'] / (1.0 + np.exp((np.log10(data['tilde_Lambda']) - p['lambda0']) / w_value))
    
    scatter = ax2.scatter(data['tilde_Lambda'], data['f_obs'], 
                         c=np.abs(eos_corr_data)*100, cmap='viridis', 
                         alpha=0.6, s=50)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('|EOS Correction| [%]')
    
    ax2.set_xlabel('$\\tilde{\\Lambda}$', fontsize=12)
    ax2.set_ylabel('$f_2$ [Hz]', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_title('Data Distribution by EOS Correction', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_4_eos_correction.png'), dpi=300)
    plt.close()

def fig_5_corner_plot(data, params, output_dir):
    """Figure 5: Corner plot showing parameter constraints"""
    # Check if we have actual MCMC samples
    mcmc_samples_file = os.path.join(os.path.dirname(output_dir), 'results', 'mcmc_samples_M2.npy')
    
    if not os.path.exists(mcmc_samples_file):
        print("Warning: MCMC samples not found. Skipping corner plot.")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'Corner plot requires MCMC samples\nRun 01_calibrate_model.py first', 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'fig_5_corner_plot.png'), dpi=300)
        plt.close()
        return
    
    # Load actual MCMC samples
    print("Loading actual MCMC samples for corner plot...")
    samples = np.load(mcmc_samples_file)
    
    labels = ['$\\alpha$', '$\\beta$', '$\\delta$', '$\\log_{10}(\\Lambda_0)$', '$w$']
    
    # Import corner with error handling
    try:
        import corner
    except ImportError:
        print("Warning: corner package not installed. Skipping corner plot.")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'Corner plot requires "corner" package\nInstall with: pip install corner', 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'fig_5_corner_plot.png'), dpi=300)
        plt.close()
        return
    
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        smooth=1.5,
        bins=30
    )
    
    fig.suptitle('Parameter Constraints', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_5_corner_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def fig_6_kfold_cv(data, params, output_dir):
    """Figure 6: k-fold cross-validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    plt.figure(figsize=(8, 8))
    
    all_train_pred = []
    all_train_obs = []
    all_test_pred = []
    all_test_obs = []
    
    for i, (train_idx, test_idx) in enumerate(kf.split(data['f_obs'])):
        # Fit on training data
        popt, _ = curve_fit(
            f_model_optimal,
            (data['f_base'][train_idx], data['q'][train_idx], 
             data['tilde_Lambda'][train_idx]),
            data['f_obs'][train_idx],
            sigma=data['f_err'][train_idx],
            absolute_sigma=True,
            p0=INITIAL_PARAMS_M2_FIXED_W + [0.1],  # Add w=0.1 for M2 full model
            bounds=([0.1, -10, -1, 2, 0.05], [10, 10, 1, 3.5, 1])
        )
        
        # Predict on train and test
        train_pred = f_model_optimal(
            (data['f_base'][train_idx], data['q'][train_idx], 
             data['tilde_Lambda'][train_idx]), *popt)
        test_pred = f_model_optimal(
            (data['f_base'][test_idx], data['q'][test_idx], 
             data['tilde_Lambda'][test_idx]), *popt)
        
        all_train_pred.extend(train_pred)
        all_train_obs.extend(data['f_obs'][train_idx])
        all_test_pred.extend(test_pred)
        all_test_obs.extend(data['f_obs'][test_idx])
    
    # Plot
    plt.scatter(all_train_obs, all_train_pred, alpha=0.5, label='Training', s=30)
    plt.scatter(all_test_obs, all_test_pred, alpha=0.7, color='red', 
                label='Test', s=50, edgecolors='black')
    
    # Perfect correlation
    min_f = min(min(all_train_obs), min(all_test_obs))
    max_f = max(max(all_train_obs), max(all_test_obs))
    plt.plot([min_f, max_f], [min_f, max_f], 'k--', lw=2)
    
    plt.xlabel('$f_{2,NR}$ [Hz]', fontsize=12)
    plt.ylabel('$f_{2,predicted}$ [Hz]', fontsize=12)
    plt.title('5-Fold Cross-Validation', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_6_kfold_cv.png'), dpi=300)
    plt.close()

def fig_7_fit_comparison(data, params, output_dir):
    """Figure 7: Comparison with literature"""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(data['tilde_Lambda'], data['f_obs'], alpha=0.5, s=30, 
                label='CoRe simulations')
    
    # Our optimal model predictions
    tilde_lambda_range = np.logspace(np.log10(data['tilde_Lambda'].min()), 
                                    np.log10(data['tilde_Lambda'].max()), 100)
    median_q = np.median(data['q'])  # Median mass ratio
    # Calculate effective radius for median total mass of 2.7 M_sun
    R_L = tilde_lambda_range**(1/5) * (G * (2.7 * M_sun)) / c**2  # [meters]
    # Base Keplerian frequency [Hz]
    f_base_range = np.sqrt(G * (2.7 * M_sun) / R_L**3) / (2 * np.pi)
    
    p = params['parameters']
    # Handle both M2 and M2_fixed_w models
    if 'w' in p:
        f_our = f_model_optimal(
            (f_base_range, np.full_like(f_base_range, median_q), tilde_lambda_range),
            p['alpha'], p['beta'], p['delta'], p['lambda0'], p['w']
        )
    else:
        # M2_fixed_w uses fixed w=0.1
        f_our = f_model_optimal(
            (f_base_range, np.full_like(f_base_range, median_q), tilde_lambda_range),
            p['alpha'], p['beta'], p['delta'], p['lambda0']
        )
    
    plt.plot(tilde_lambda_range, f_our, 'r-', lw=2, 
             label='This work (M2)')
    
    # Bauswein 2012 approximation
    f_bauswein = (2.83 - 0.375 * np.log10(tilde_lambda_range)) * 1000
    plt.plot(tilde_lambda_range, f_bauswein, 'b--', lw=2, 
             label='Bauswein et al. 2012')
    
    plt.xlabel('$\\tilde{\\Lambda}$', fontsize=12)
    plt.ylabel('$f_2$ [Hz]', fontsize=12)
    plt.xscale('log')
    plt.title('Comparison with Literature', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_7_fit_comparison.png'), dpi=300)
    plt.close()

def fig_8_model_comparison(output_dir):
    """Figure 8: Model comparison using AIC/BIC"""
    # Load model comparison results
    comparison_file = "./results/model_comparison.json"
    
    if not os.path.exists(comparison_file):
        print("Warning: model_comparison.json not found, skipping fig_8")
        return
        
    with open(comparison_file, 'r') as f:
        results = json.load(f)
    
    models = ['M0', 'M1', 'M2', 'M2_fixed_w', 'M3', 'M_full']
    aic_values = []
    bic_values = []
    k_values = []
    
    for m in models:
        if m in results:
            aic_values.append(results[m]['aic'])
            bic_values.append(results[m]['bic'])
            k_values.append(len(results[m]['params']))
        else:
            # Skip if model not in results
            models.remove(m)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AIC comparison
    ax1.bar(models, aic_values, color='steelblue', alpha=0.7)
    ax1.axhline(min(aic_values), color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('AIC', fontsize=12)
    ax1.set_title('Model Comparison: AIC', fontsize=14)
    
    # Add k values on bars
    for i, (model, aic, k) in enumerate(zip(models, aic_values, k_values)):
        ax1.text(i, aic + 5, f'k={k}', ha='center', va='bottom')
    
    # BIC comparison
    ax2.bar(models, bic_values, color='darkgreen', alpha=0.7)
    ax2.axhline(min(bic_values), color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('BIC', fontsize=12)
    ax2.set_title('Model Comparison: BIC', fontsize=14)
    
    # Add k values on bars
    for i, (model, bic, k) in enumerate(zip(models, bic_values, k_values)):
        ax2.text(i, bic + 5, f'k={k}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_8_model_comparison.png'), dpi=300)
    plt.close()

def fig_9_detector_sensitivity(data, params, output_dir):
    """Figure 9: Detector sensitivity curves"""
    plt.figure(figsize=(10, 6))
    
    # Frequency range
    f = np.logspace(1, 4, 1000)
    
    # Advanced LIGO design sensitivity
    # Simplified analytical fit to design sensitivity curve
    # Best sensitivity around 100-300 Hz
    f_min = 20  # Hz
    f0 = 150  # Hz - optimal frequency
    
    # Build piecewise sensitivity curve for aLIGO
    aLIGO = np.zeros_like(f)
    for i, freq in enumerate(f):
        if freq < f_min:
            aLIGO[i] = 1e-20  # Wall at low frequency
        elif freq < 50:
            aLIGO[i] = 1e-22 * (50/freq)**2
        elif freq < 200:
            aLIGO[i] = 3e-23  # Best sensitivity region
        elif freq < 1000:
            aLIGO[i] = 3e-23 * (freq/200)**0.5
        else:
            aLIGO[i] = 1e-22 * (freq/1000)**2
    
    # Einstein Telescope - roughly 10x better
    ET = aLIGO / 10
    
    # Make sure values are in reasonable range
    aLIGO = np.clip(aLIGO, 1e-24, 1e-20)
    ET = np.clip(ET, 1e-25, 1e-21)
    
    plt.loglog(f, aLIGO, 'b-', lw=2, label='Advanced LIGO')
    plt.loglog(f, ET, 'g-', lw=2, label='Einstein Telescope')
    
    # Mark f2 range
    plt.axvspan(2000, 4000, alpha=0.2, color='red', label='$f_2$ range')
    
    # Mark specific f2 for median system
    median_f2 = np.median(data['f_obs'])
    plt.axvline(median_f2, color='red', linestyle='--', lw=2,
                label=f'Typical $f_2$ = {median_f2:.0f} Hz')
    
    # Add annotation for f2 detection challenge
    plt.text(3000, 5e-23, '$f_2$ detection\nchallenging', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             ha='center', fontsize=10)
    
    plt.xlabel('Frequency [Hz]', fontsize=12)
    plt.ylabel('Strain Sensitivity [Hz$^{-1/2}$]', fontsize=12)
    plt.title('Detector Sensitivity and Post-merger Frequencies', fontsize=14)
    plt.xlim(10, 10000)
    plt.ylim(1e-24, 1e-20)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_9_detector_sensitivity.png'), dpi=300)
    plt.close()

def main():
    """Generate all figures"""
    output_dir = "./figs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data and parameters...")
    df, params = load_data_and_params()
    data = prepare_data(df)
    
    print("\nGenerating figures...")
    
    print("  Figure 1: Calibration scatter...")
    fig_1_calibration_scatter(data, params, output_dir)
    
    print("  Figure 2: Residuals histogram...")
    fig_2_residuals_hist(data, params, output_dir)
    
    print("  Figures 3a-c: Systematic checks...")
    fig_3_residuals_systematics(data, params, output_dir)
    
    print("  Figure 4: EOS correction analysis...")
    fig_4_eos_correction_analysis(data, params, output_dir)
    
    print("  Figure 5: Corner plot...")
    fig_5_corner_plot(data, params, output_dir)
    
    print("  Figure 6: Cross-validation...")
    fig_6_kfold_cv(data, params, output_dir)
    
    print("  Figure 7: Literature comparison...")
    fig_7_fit_comparison(data, params, output_dir)
    
    print("  Figure 8: Model comparison...")
    fig_8_model_comparison(output_dir)
    
    print("  Figure 9: Detector sensitivity...")
    fig_9_detector_sensitivity(data, params, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")

if __name__ == "__main__":
    main()