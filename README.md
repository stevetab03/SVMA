# AEM-SVMA: Asymptotic Equilibrium Model for Options Microstructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Math: SDE & Generator Theory](https://img.shields.io/badge/Math-SDE_%7C_Generator_Theory-blueviolet)]()
[![Methods: Global/Local Optimization](https://img.shields.io/badge/Methods-Differential_Evolution_%7C_L--BFGS--B-success)]()
[![ML: LSTM Benchmark](https://img.shields.io/badge/ML-LSTM_Benchmark-orange)]()

**Author:** Liyuan Zhang
**Status:** R&D Paused / Portfolio Showcase

---

## Executive Summary

Traditional financial models — Black-Scholes, Heston, GARCH — are built around
price dynamics in isolation. They do not account for the structural forces that
govern modern index and ETF markets: dealer gamma hedging, localized open-interest
clustering, and strike pinning. These are not noise. They are the dominant
microstructure of options markets today.

The **Asymptotic Equilibrium Model (AEM)** and its core engine, the
**Stochastic Volatility Modulated Asymptotic (SVMA)** system, bridge structural
stochastic calculus and observable market-flow signals. The central architectural
contribution is a modular drift decomposition that gates volatility and price
dynamics through statistically validated activations, while proving that the system
converges to a unique stationary distribution under asymptotic conditions.

Full derivations, theorem statements, and proofs are documented in
`theory/svma_monograph.pdf`.

---

## What the Model Does

- Models the joint dynamics of spot price, open-interest-anchored strike, and
  stochastic volatility as a multi-dimensional coupled system
- Decomposes the price drift into four distinct structural forces: baseline mean
  reversion, VECM correction, localized strike pinning, and
  momentum — each activated only when statistically warranted
- Proves via infinitesimal generator analysis that the system admits a unique
  stationary measure, preventing numerical instability during high-volatility regimes
- Calibrates against observed option surfaces using a vega-weighted objective function
  that prioritizes fidelity where risk exposure is largest

---

## Drift Decomposition and Activations

The architectural innovation is in how the price drift is structured. Rather than
a constant or affine function, the drift is a modular sum of forces that activate
conditionally on market state:

- **Baseline Mean Reversion** pulls price toward the open-interest-anchored strike
as the structural equilibrium.
- **VECM Correction** ensures the correction term is applied only when the statistical relationship is valid — not at all times.
- **Local Pinning** applies a kernel-weighted attraction toward high-OI
strikes, modulated by the VegEx Ratio. 
- **Momentum** captures short-horizon directional persistence, gated by
realized flow signals.

The gating design means the model is not always-on. Activations earn their
inclusion through statistical tests on each window. This is the key distinction
from standard affine SDE models.

---

## Theoretical Grounding

The framework is grounded in the infinitesimal generator of the SVMA process,
which characterizes the full dynamics in operator form. For a test function $f \in C^{1,2}$, the generator is defined as:

$$ \mathcal{A}f = \frac{\partial f}{\partial t} + \mu_F \frac{\partial f}{\partial F} + \mu_K \frac{\partial f}{\partial K} + \mu_G \frac{\partial f}{\partial G} + \frac{1}{2} \left[ G^2 \frac{\partial^2 f}{\partial F^2} + \sigma_K^2 \frac{\partial^2 f}{\partial K^2} + \nu^2 G^2 \frac{\partial^2 f}{\partial G^2} \right] $$

This leads to the **Backward Kolmogorov Equation**, which provides the deterministic characterization of conditional expectations required for out-of-sample forecasting:

$$ \frac{\partial u}{\partial s} + \mathcal{A}_s u = 0, \quad u(t, x) = g(x) $$

By satisfying the **SVMA Ergodicity and Convergence Conditions (SECC)**, we prove that the generator
satisfies the SVMA Ergodicity and Convergence Conditions (SECC) establishes:

- **Existence and uniqueness** of a stationary distribution
- **Variance stabilization** — volatility remains bounded under the gating structure
- **Asymptotic equilibrium** — the system returns to a well-defined distributional
  steady state after perturbation

These results are not empirical observations. They are theorems derived from the
structure of the model. The proofs are in `theory/svma_monograph.pdf`.

---

## Engineering Pipeline

### Calibration (`calibrator.py`)

Calibration against non-convex option surfaces uses a two-stage strategy minimising
a vega-weighted price RMSE objective. Vega-weighting concentrates calibration
accuracy where dealer exposure — and therefore model risk — is greatest.

- **Stage 1 (Global):** Differential Evolution explores the full parameter space
  without gradient information, avoiding local minima on the non-convex surface.
- **Stage 2 (Local):** L-BFGS-B refines the DE output to gradient-level accuracy
  within the stability boundaries imposed by SECC.

### Simulation (`sde.py`)

Milstein scheme with operator splitting. The splitting separates piecewise-constant
activation evaluation from continuous stochastic diffusion, preserving path accuracy
and preventing activation discontinuities from contaminating the diffusion step.

### Validation (`validation.py`)

ADF stationarity tests on cointegration residuals, VegEx ratio computation, and
out-of-sample vega-weighted RMSE against benchmark models.

---

## Model Benchmarking

Benchmarked against standard parametric and non-parametric architectures on
2023–2025 options data from Polygon.io.

| Model | Avg. Vega-Weighted RMSE | vs. SVMA |
|---|---|---|
| **SVMA (Full Framework)** | **25.56%** | — |
| LSTM (Deep Learning) | 32.71% | −21.8% |
| Heston Model | 37.87% | −32.5% |
| Black-Scholes (RV) | 50.38% | −49.3% |
| GARCH(1,1) | 50.83% | −49.7% |

SVMA outperforms LSTM by 21.8% on vega-weighted RMSE — a structurally grounded
SDE system outperforming a deep learning baseline on the same data. The
interpretation: when the structural forces are real and identifiable, encoding
them explicitly outperforms learning them implicitly from data.

---

## Retrospective: Descriptive vs. Predictive Power

The project is currently paused following a rigorous empirical audit.

**Descriptive success.** The AEM-SVMA framework consistently outperformed LSTM
in fitting the volatility surface. The structural SDE approach proved to be a
superior characterisation of market microstructure state and dealer rebalancing
pressure. It captures where the market is with higher fidelity than black-box
weights.

**Predictive reality.** The model is not a forecasting tool in the conventional
sense. It identifies where price should return under equilibrium, but cannot
guarantee the timing of that return. Exogenous shocks — earnings, macro data,
geopolitical events — are outside the model's scope by design.

**Final takeaway.** A theoretically grounded, activation-driven SDE system can
outperform deep learning for high-fidelity microstructure modeling. Generating
risk-free arbitrage remains mathematically elusive, but that was never the claim.
The claim was that structure beats agnosticism when the structure is real.
The benchmarks support that claim.

---

## Mathematical Documentation

Full derivations of the four core theoretical results are available in
`theory/svma_monograph.pdf`:

- SVMA Ergodicity and Convergence Conditions (SECC)
- SVMA Fundamental Asymptotic Equilibrium Theorem
- SVMA Variance Stabilization Lemma
- SVMA Equilibrium Distribution Theorem

The monograph is available in compiled PDF form. The LaTeX source is not
distributed publicly.

---

## Related Work

- [stevetab03/ORBIT](https://github.com/stevetab03/ORBIT) — applies the same
  structural SDE philosophy to WTI futures-spot basis convergence, with an
  ML enhancement layer benchmarking Neural SDE, Bayesian Optimization, and
  LSTM against the classical framework.

---

## Contact

**Monograph:** Available upon request for full derivations of my SVMA Asymptotic Theorems i.e., SVMA Ergodicity and Convergence Conditions, SVMA Fundamental Asymptotic Equilibrium Theorem, SVMA Variance Stabilization Lemma, and SVMA Equilibrium Distribution Theorem.  
**LinkedIn:** https://www.linkedin.com/in/hlzhang/  
**GitHub:** https://github.com/stevetab03
