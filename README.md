# Logistic Correction to the Universal Post-Merger f₂ Law
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](LICENSE)

**Fast, differentiable Python implementation of the logistic-extended universal relation  
$f_2(q,\tilde{\Lambda})$ for binary-neutron-star post-merger spectra.**

> *“A single smooth equation instead of piece-wise fits — ready for Einstein Telescope analysis pipelines.”*

---

## Why this matters

* **Smooth across the “phase-transition knee”.** The logistic term bridges soft- and stiff-EOS regimes around $\tilde{\Lambda}_{\text{crit}}\!\approx\!1216$.
* **Calibrated on 221 numerical-relativity simulations** (146 CoRe + 75 new runs) processed with a unified Welch–ψ₄ pipeline.
* **Instant drop-in for Bayesian inference.** Call one function, get $f_2$ and analytic gradients for autodiff.
* **Reproducible.** All data, notebooks and CI tests included; each release is archived on Zenodo with a citable DOI.

---

## Quick start

