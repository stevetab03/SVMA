# AEM-SVMA: Asymptotic Equilibrium Model for Options Microstructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Math: SDE & PDE Theory](https://img.shields.io/badge/Math-SDE_%7C_Generator_Theory-blueviolet)]()
[![Methods: Global/Local Optimization](https://img.shields.io/badge/Methods-Differential_Evolution_%7C_L--BFGS--B-success)]()

**Author:** Liyuan Zhang  
**Status:** R&D Paused / Portfolio Showcase  

## Executive Summary
Traditional financial models like Geometric Brownian Motion (Black-Scholes) and affine variance models (Heston) systematically fail to account for the "gravity" of options market microstructure. Today’s index and leveraged ETFs (SPY, GUSH) are governed by dealer gamma hedging, localized open-interest (OI) clusters, and strike pinning.

The **Asymptotic Equilibrium Model (AEM)** and its core engine, the **Stochastic Volatility Modulated Asymptotic (SVMA)** system, is a framework designed to bridge the gap between structural stochastic calculus and observable market-flow signals. By utilizing a "Drift Decomposition" approach, the model modulates volatility and price-action via statistically gated activations, ensuring the system remains anchored to a theoretically proven **Asymptotic Equilibrium.**

---

## The Mathematical Engine

### Core SDE System
The SVMA system operates on a 3D state space tracking the underlying spot price ($F_t$), the dynamic open-interest-anchored strike ($K_t$), and the stochastic volatility factor ($G_t$). Unlike the Heston convention, we model the volatility *level* directly to simplify the diffusion scale:

$$ dF_t = \mu_F(X_t, t) dt + G_t dW^F_t $$
$$ dK_t = \kappa(k^*_t - K_t) dt + \sigma_K dW^K_t $$
$$ dG_t = \mu(K_t - G_t) dt + \nu G_t dW^G_t $$

### Drift Decomposition & Activations
The architectural innovation lies in $\mu_F$, where the drift is not a constant, but a modular sum of baseline forces and **statistically gated activations**:

$$ \mu_F = \underbrace{\alpha_1(K_t - F_t)}_{\text{Baseline Mean Reversion}} + \underbrace{\Phi_{\omega,\delta}(t)\gamma(F_t - \lambda K_t)}_{\text{VECM Cointegration Gate}} + \underbrace{\Psi(F_t, k^*_t) M_{\text{flow}}(t)}_{\text{Local Pinning force}} + \underbrace{\Xi_{\text{mom}}(t)}_{\text{Momentum}} $$

1.  **VECM Activation:** Gated by Augmented Dickey-Fuller (ADF) tests to confirm stationarity in cointegration residuals.
2.  **Pinning Activation:** A localized kernel centered at high-OI strikes ($k^*$), modulated by the **VegEx Ratio** (Vega Exposure relative to Gamma Exposure).

---

## Infinitesimal Generator & PDE Framework

To ensure the model’s reliability for risk management (VaR) and pricing, the framework is grounded in the **Infinitesimal Generator** $\mathcal{A}$ of the SVMA process. For a test function $f \in C^{1,2}$, the generator is defined as:

$$ \mathcal{A}f = \frac{\partial f}{\partial t} + \mu_F \frac{\partial f}{\partial F} + \mu_K \frac{\partial f}{\partial K} + \mu_G \frac{\partial f}{\partial G} + \frac{1}{2} \left[ G^2 \frac{\partial^2 f}{\partial F^2} + \sigma_K^2 \frac{\partial^2 f}{\partial K^2} + \nu^2 G^2 \frac{\partial^2 f}{\partial G^2} \right] $$

This leads to the **Backward Kolmogorov Equation**, which provides the deterministic characterization of conditional expectations required for out-of-sample forecasting:

$$ \frac{\partial u}{\partial s} + \mathcal{A}_s u = 0, \quad u(t, x) = g(x) $$

By satisfying the **SVMA Ergodicity and Convergence Conditions (SECC)**, we prove that the system admits a unique stationary measure $\pi$, preventing numerical "explosions" during high-volatility regimes.

---

## Engineering & Optimization Pipeline

### Global/Local Hybrid Calibration
Calibration against non-convex option surfaces is handled by a robust two-stage optimizer minimizing a **Vega-Weighted Price RMSE** objective:
*   **Stage 1 (Global):** **Differential Evolution (DE)** explores the parameter space to avoid local minima.
*   **Stage 2 (Local):** **L-BFGS-B** refines the DE output to reach gradient-level accuracy within SECC-enforced stability boundaries.

### Numerical Integration
*   **Method:** Milstein Scheme with Operator Splitting.
*   **Design:** Separates piecewise-constant activation evaluation from continuous stochastic diffusion, ensuring path accuracy $O(\Delta t)$ superior to standard Euler-Maruyama.

---

## Model Benchmarking
The model was rigorously benchmarked against standard parametric and non-parametric architectures using 2023–2025 options data from Polygon.io.

| Model | Avg. Vega-Weighted RMSE | Improvement vs. SVMA |
| :--- | :--- | :--- |
| **SVMA (Full Framework)** | **25.56%** | **--** |
| **LSTM (Deep Learning)** | 32.71% | -21.8% |
| **Heston Model** | 37.87% | -32.5% |
| **Black-Scholes (RV)** | 50.38% | -49.3% |
| **GARCH(1,1)** | 50.83% | -49.7% |

---

## Retrospective: Descriptive vs. Predictive Power

The project is currently on pause following a deep empirical audit. 

*   **Descriptive Success:** The AEM-SVMA framework **consistently outperformed LSTM Deep Learning baselines** in fitting the volatility surface. The structural SDE approach proved to be a superior "describer" of market state and dealer rebalancing pressure. It effectively captures "where the market is" better than black-box weights.
*   **Predictive Reality:** While the model provides an exceptional "map" of microstructure, it is not a "crystal ball." Because the market is dominated by exogenous shocks and unpredictable stochasticity, the model can identify *where price should return* under equilibrium, but it cannot guarantee the *timing* of that return. 
*   **Final Takeaway:** This project validates that a theoretically-grounded, activation-driven system can outperform ML for high-fidelity modeling, even if generating risk-free arbitrage remains mathematically elusive.

---

## Contact:

**LinkedIn:** https://www.linkedin.com/in/hlzhang/  
**GitHub:** https://github.com/stevetab03  
**Thesis/Monograph:** Available upon request for full derivations of SECC, SFAET, and SEDT theorems.
