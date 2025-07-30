# Code for "Universal Law for Post-Merger Gravitational Waves from Binary Neutron Star Mergers"

This repository contains the Python scripts used to generate the results and figures for the paper "Universal Law for Post-Merger Gravitational Waves from Binary Neutron Star Mergers".

**DOI:** [10.5281/zenodo.16611969](https://doi.org/10.5281/zenodo.16611969)

## Abstract

The dominant post-merger gravitational-wave frequency, $f_2$, from binary neutron star (BNS) mergers follows a universal scaling law that connects it to the tidal deformability parameter, $\tilde{\Lambda}$, measured during the inspiral phase. This work presents a simple, physically motivated, and highly accurate universal law:

$$ f_{2}=\alpha\,\sqrt{\dfrac{G\,M_{\text{tot}}}{R_{\Lambda}^{3}}}\;\bigl[1+\beta\,(q-1)^{2}\bigr],\qquad R_{\Lambda}=\bigl(\tilde\Lambda\bigr)^{1/5}\,\dfrac{G M}{c^{2}} $$

where $M_{\text{tot}}$ is the total mass, $q$ is the mass ratio, and $(\alpha, \beta)$ are universal constants calibrated from numerical relativity simulations. This law provides a powerful tool for multi-messenger astronomy, enabling a self-consistent check of the neutron star equation of state (EOS) from inspiral to post-merger.

## Scripts

The scripts are organized as follows:

- **`fit_universal_law.py`**: This is the main script for fitting the universal law parameters $(\alpha, \beta)$ to numerical relativity (NR) data. It loads simulation data, performs a non-linear least-squares fit, and generates diagnostic plots to assess the quality of the fit. It also calculates key accuracy metrics like mean and maximum relative errors.

- **`bayes_inverter.py`**: This script demonstrates the practical application of the universal law. It implements a Bayesian framework to show how an observed $f_2$ frequency can be used to improve the constraints on the tidal deformability $\tilde{\Lambda}$. It simulates the expected improvement in measurement precision for LIGO/Virgo/KAGRA at O5 sensitivity.

- **`plot_theoretical_law.py`**: This script visualizes the theoretical properties of the universal law. It generates plots showing how the $f_2$ frequency depends on the key physical parameters: total mass ($M_{\text{tot}}$), mass ratio ($q$), and tidal deformability ($\tilde{\Lambda}$).

- **`plot_universal_scaling_q1.py`**: This script generates a plot to demonstrate the "universality" of the scaling law for the equal-mass ($q=1$) case. It shows that data from various equations of state (EOS) collapse onto a single horizontal line when the frequency is normalized by the characteristic gravitational frequency, validating the core assumption of the model.

- **`plot_mass_ratio_correction.py`**: This script visualizes the quadratic correction term for unequal-mass mergers. It shows that the relative shift in the $f_2$ frequency is well-described by a quadratic function of $(1-q)$, justifying the inclusion of the $\beta(1-q)^2$ term in the law.

- **`plot_fit_diagnostics.py`**: This script generates a comprehensive set of diagnostic plots to evaluate the performance of the best-fit model. This includes plots of residuals versus various parameters, a Q-Q plot to check for normality, and a histogram of relative errors.

## How to Run

To regenerate the figures from the paper, you can run the scripts directly. Ensure you have the required Python libraries installed (`numpy`, `scipy`, `matplotlib`, `seaborn`).

```bash
python fit_universal_law.py
python bayes_inverter.py
python plot_theoretical_law.py
# ... and so on for the other plotting scripts
```

The scripts will generate figures in the `../figs/` directory.

```
