#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Central configuration file for M2_fixed_w model parameters
All scripts should import parameters from this file to ensure consistency
"""

# --- Model Selection Flags ---
USE_ZETA_CORRECTION = False  # Set to False to run without systematic correction

# Reference model parameters from model_comparison.json
M2_FIXED_W_PARAMS = {
    'alpha': 4.322,
    'beta': -0.784,
    'delta': -0.512,
    'lambda0': 3.085,  # log10(lambda0), actual lambda0 = 1217
    'w': 0.1  # Fixed width parameter
}

# Derived values
LAMBDA0_VALUE = 10**M2_FIXED_W_PARAMS['lambda0']  # 1217

# Error scaling factor - REMOVED per reviewer request
# Artificial scaling to force χ²/dof = 1 is methodologically incorrect
# ERROR_SCALING_FACTOR = 0.997  # DEPRECATED - do not use
ERROR_SCALING_FACTOR = 1.0  # Use physical errors without scaling

# Reference statistics
REFERENCE_CHI2_DOF = 0.994
REFERENCE_AIC = 149.1
REFERENCE_BIC = 161.0

# Initial parameters for optimization (when needed)
INITIAL_PARAMS_M2_FIXED_W = [
    M2_FIXED_W_PARAMS['alpha'],
    M2_FIXED_W_PARAMS['beta'],
    M2_FIXED_W_PARAMS['delta'],
    M2_FIXED_W_PARAMS['lambda0']
]

# Parameter bounds for M2_fixed_w model
BOUNDS_M2_FIXED_W = ([0.1, -10, -1, 2], [10, 10, 1, 3.5])


def predict_f2(f_base, q, tilde_lambda, alpha=None, beta=None, delta=None, lambda0=None, w=None, zeta=None, code_indicator=None):
    """
    Universal f2 prediction formula
    Currently implements M2_fixed_w model with optional systematic correction
    
    Parameters:
    -----------
    f_base : array-like
        Base frequency from Newtonian scaling [Hz]
    q : array-like
        Mass ratio (m2/m1, where m1 >= m2)
    tilde_lambda : array-like
        Effective tidal deformability
    alpha, beta, delta, lambda0, w : float, optional
        Model parameters. If not provided, uses reference values from M2_FIXED_W_PARAMS
    zeta : float, optional
        Systematic correction parameter for different codes (BAM vs THC)
    code_indicator : array-like, optional
        Indicator for code type (+1 for THC, -1 for BAM)
        
    Returns:
    --------
    f2 : array-like
        Predicted post-merger frequency [Hz]
        
    Notes:
    ------
    Current model: M2_fixed_w with optional systematic correction
    f2 = α * f_base * (1 + β*(q-1)²) * (1 + δ/(1 + exp((log10(Λ̃) - λ₀)/w))) * (1 + ζ*P_code)
    
    Model validity range:
    - Λ̃: 200 - 3000 (reliable), 117 - 3500 (with warnings)
    - q: 0.7 - 1.0
    - M_tot: 2.2 - 3.2 M_sun
    """
    import numpy as np
    import warnings
    
    # Use default parameters if not provided
    if alpha is None:
        alpha = M2_FIXED_W_PARAMS['alpha']
    if beta is None:
        beta = M2_FIXED_W_PARAMS['beta']
    if delta is None:
        delta = M2_FIXED_W_PARAMS['delta']
    if lambda0 is None:
        lambda0 = M2_FIXED_W_PARAMS['lambda0']
    if w is None:
        w = M2_FIXED_W_PARAMS['w']
    
    # Check validity ranges and issue warnings
    tilde_lambda_scalar = np.atleast_1d(tilde_lambda)
    q_scalar = np.atleast_1d(q)
    
    # Check Λ̃ range
    if np.any(tilde_lambda_scalar < 117):
        warnings.warn(f"Model extrapolation: Λ̃ < 117 detected (min={np.min(tilde_lambda_scalar):.1f}). "
                     "Predictions may exceed 6 kHz and be unreliable.", stacklevel=2)
    elif np.any(tilde_lambda_scalar < 200):
        warnings.warn(f"Model extrapolation: Λ̃ < 200 detected (min={np.min(tilde_lambda_scalar):.1f}). "
                     "Use with caution.", stacklevel=2)
    
    if np.any(tilde_lambda_scalar > 3500):
        warnings.warn(f"Model extrapolation: Λ̃ > 3500 detected (max={np.max(tilde_lambda_scalar):.1f}). "
                     "Limited calibration data in this regime.", stacklevel=2)
    
    # Check q range
    if np.any(q_scalar < 0.7):
        warnings.warn(f"Model extrapolation: q < 0.7 detected (min={np.min(q_scalar):.2f}). "
                     "Model not calibrated for extreme mass ratios.", stacklevel=2)
    
    # Mass ratio correction
    f_model = alpha * f_base * (1.0 + beta * (q - 1)**2)
    
    # EOS correction (logistic function)
    delta_eos = delta / (1.0 + np.exp((np.log10(tilde_lambda) - lambda0) / w))
    
    # Final prediction without systematic correction
    f2 = f_model * (1.0 + delta_eos)
    
    # Apply systematic correction if provided
    if zeta is not None and code_indicator is not None:
        f2 = f2 * (1.0 + zeta * code_indicator)
    
    return f2


def calculate_aic_bic_likelihood(y_obs, y_pred, y_err, n_params):
    """
    Calculate AIC and BIC using full log-likelihood
    
    Parameters:
    -----------
    y_obs : array
        Observed values
    y_pred : array
        Predicted values
    y_err : array
        Error values
    n_params : int
        Number of model parameters
        
    Returns:
    --------
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    chi2 : float
        Chi-squared value
        
    Notes:
    ------
    Full formulas with log-likelihood:
    log(L) = -0.5 * [χ² + n*log(2π) + Σlog(σᵢ²)]
    AIC = 2k - 2log(L)
    BIC = k*ln(n) - 2log(L)
    """
    import numpy as np
    
    n_data = len(y_obs)
    chi2 = np.sum(((y_obs - y_pred) / y_err)**2)
    
    # Log-likelihood for Gaussian errors
    log_likelihood = -0.5 * (chi2 + n_data * np.log(2 * np.pi) + 
                            np.sum(np.log(y_err**2)))
    
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_data) - 2 * log_likelihood
    
    return aic, bic, chi2