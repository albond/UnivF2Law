# Logistic Correction to the Universal Post-Merger f₂ Law
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16611969.svg)](https://doi.org/10.5281/zenodo.16611969)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

**Fast, differentiable Python implementation of the logistic-extended universal relation  
$f_2(q,\tilde{\Lambda})$ for binary-neutron-star post-merger spectra.**

> *“A single smooth equation instead of piece-wise fits — ready for Einstein Telescope analysis pipelines.”*

![BNS Merger Animation](./bns_merger_enhanced.gif)

---

## Why this matters

* **Smooth across the “phase-transition knee”.** The logistic term bridges soft- and stiff-EOS regimes around $\tilde{\Lambda}_{\text{crit}}\!\approx\!1216$.
* **Calibrated on 221 numerical-relativity simulations** (146 CoRe + 75 new runs) processed with a unified Welch–ψ₄ pipeline.
* **Instant drop-in for Bayesian inference.** Call one function, get $f_2$ and analytic gradients for autodiff.
* **Reproducible.** All data, notebooks and CI tests included; each release is archived on Zenodo with a citable DOI.

---

## Quick start

### Prerequisites

- Python 3.9+
- ~10 GB disk space for CoRe simulation data
- Install dependencies:
```bash
pip install -r requirements.txt
```
- To ensure the project runs smoothly, please create the following subdirectories at the root of the repository. These folders are required for input/output during data processing and model execution:
<pre>
<project-root>/
├── CoRe/       # Downloaded HDF5 files from numerical simulations
├── data/       # Processed simulation data
├── figs/       # All publication-ready figures
├── results/    # Model parameters, metrics, and statistical analyses
</project-root>
</pre>

### Step 1: Data Preparation

Download and process numerical relativity simulations from CoRe and other catalogs:

```bash
# 1.1 Collect metadata from CoRe database
python data_prep_01_select_core_simulations.py

# 1.2 Download HDF5 simulation files (~8GB)
python data_prep_02_download_core_hdf5.py
# Downloads: CoRe/BAM_*/data.h5, CoRe/THC_*/data.h5

# 1.3 Extract post-merger frequencies using Welch-ψ₄ method
python data_prep_03_extract_f2_from_hdf5.py
# Creates: core_f2_frequencies.csv

# 1.4 Combine all catalogs (CoRe + literature data)
python data_prep_04_download_all_catalogs.py
# Creates: nr_simulations_with_f2.csv
```

### Step 2: Model Analysis & Figures

Reproduce all results and figures from the paper:

```bash
# 2.1 Calibrate the logistic-extended model using MCMC
python 01_calibrate_model.py
# Creates Results: results/model_comparison.json, results/optimal_model.json, results/mcmc_chains.h5
# Creates Figures: figs/f2_extracted_summary.png, figs/mcmc_diagnostics.png
# Runtime: ~5 min

# 2.2 Generate all paper figures
python 02_generate_figures.py
# Creates: figs/fig_1_calibration_scatter.png
#          figs/fig_2_residuals_hist.png
#          figs/fig_3a_residuals_vs_Lambda.png
#          figs/fig_3b_residuals_vs_q.png
#          figs/fig_3c_residuals_vs_Mtot.png
#          figs/fig_4_eos_correction.png
#          figs/fig_5_corner_plot.png
#          figs/fig_6_kfold_cv.png
#          figs/fig_7_fit_comparison.png
#          figs/fig_8_model_comparison.png
#          figs/fig_9_detector_sensitivity.png

# 2.3 Compare with alternative models (piecewise, polynomial)
python 03_test_alternative_models.py
# Creates: figs/fig_alternative_eos_forms.png
#          results/alternative_models.json

# 2.4 Ablation study: impact of each term
python 04_ablation_analysis.py
# Creates: figs/fig_ablation_analysis.png
#          results/ablation_analysis.json

# 2.5 Physical interpretation & EOS constraints
python 05_physical_interpretation.py
# Creates: figs/fig_physical_interpretation.png
#          results/physical_interpretation.json

# 2.6 Comprehensive error analysis
python 06_error_audit.py
# Creates: figs/fig_error_analysis.png
#          results/error_audit.json
```


### Bonus: Interactive Visualization

Create a scientifically accurate animation of binary neutron star merger:

```bash
# Generate animated GIF showing inspiral, merger, and post-merger phases
python create_merger_visual.py
# Creates: bns_merger_enhanced.gif (start from 80 MB), configurable
# Runtime: ~5-10 min
```

This visualization demonstrates key physics from our model:
- **Tidal deformation** during inspiral scaled by Λ̃ parameter
- **Post-merger oscillations** at the predicted f₂ frequency
- **Relativistic jets** with helical structure
- **Kilonova ejecta** in the equatorial plane
- **Gravitational wave emission** patterns

**Scientific Accuracy:**
- Mass ratio and tidal deformability from actual CoRe simulations
- Post-merger frequency oscillations match our calibrated f₂ model
- Deformation scales with tidal field strength proportional to Λ̃ × Mₜₒₜ / r³
- Jet structure based on GRMHD simulation results
- Color temperature evolution follows T ∝ (tₘₑᵣ𝓰ₑᵣ - t)⁻⁰·⁵

The animation parameters (q = 0.8, Mₜₒₜ = 2.7 M☉, Λ̃ = 400, f₂ = 2.8 kHz) correspond to a realistic neutron star merger scenario from our dataset.

### Results

After running all scripts, you'll have:
- **`figs/`**: All publication-ready figures
- **`results/`**: Model parameters, metrics, and statistical analyses  
- **`data/`**: Processed simulation data
- **`CoRe/`**: Downloaded HDF5 files from numerical simulations
- **`bns_merger_enhanced.gif`**: Animated visualization of the merger physics
