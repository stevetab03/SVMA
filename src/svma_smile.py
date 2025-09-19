import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

# Your calibrated SVMA parameters
ALPHA_1 = 0.080
GAMMA = 0.030
LAM = 1.000
XI = 0.150
TAU_T = -1.0

S0 = 24.78  # Current price
sigma0 = 0.456  # Realized vol

def bs_call(S, K, T, r=0.05, sigma=0.456, q=0):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol(S, K, T, market_price, r=0.05, q=0):
    def objective(sigma):
        return bs_call(S, K, T, r, sigma, q) - market_price
    
    try:
        return brentq(objective, 0.01, 2.0)
    except ValueError:
        return np.nan

def svma_option_price(tau_years, strike, n_paths=5000, r=0.05):
    dt = 1/252.0
    n_steps = int(tau_years * 252)
    
    F_paths = np.full((n_steps + 1, n_paths), S0)
    np.random.seed(42)
    
    for t in range(1, n_steps + 1):
        F_prev = F_paths[t-1]
        
        # SVMA drift with realistic mean-reversion
        forward_price = S0 * np.exp((r - 0.01) * tau_years)
        z_t = F_prev - LAM * forward_price
        drift = ALPHA_1 * (forward_price - F_prev) + GAMMA * z_t + XI * TAU_T
        
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        F_paths[t] = F_prev + drift * dt + sigma0 * F_prev * dW
    
    payoffs = np.maximum(F_paths[-1] - strike, 0)
    option_price = np.mean(payoffs) * np.exp(-r * tau_years)
    return option_price

# Generate strikes and tau
strikes = np.linspace(20, 30, 11)
tau = np.linspace(0.1, 2.0, 5)

# SVMA smile
iv_smile_svma = []
for t in tau:
    iv_line = []
    for k in strikes:
        model_price = svma_option_price(t, k, n_paths=5000)
        iv = implied_vol(S0, k, t, model_price)
        iv_line.append(iv)
    iv_smile_svma.append(iv_line)

# Black-Scholes (constant sigma)
iv_bs = []
for t in tau:
    iv_line = []
    for k in strikes:
        bs_price = bs_call(S0, k, t, sigma=sigma0)
        iv = implied_vol(S0, k, t, bs_price)
        iv_line.append(iv)
    iv_bs.append(iv_line)

# SABR (adjusted parameters for equity skew)
def sabr_iv(S, K, T, alpha=0.3, beta=0.8, rho=-0.2, nu=0.8):
    x = np.log(K/S)
    if x == 0:
        return alpha * (T ** (1 - beta))
    z = nu / alpha * ((alpha**2 * np.exp((nu**2 - 1) * T / 24)) ** (1/2)) * x
    chi = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    return alpha * (T ** (1 - beta)) * (z / chi) * np.exp((rho * nu * T) / 4)

iv_sabr = []
for t in tau:
    iv_line = []
    for k in strikes:
        iv = sabr_iv(S0, k, t)
        iv_line.append(iv)
    iv_sabr.append(iv_line)

# Plot with fixed legends
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange']
labels = ['SVMA', 'Black-Scholes', 'SABR']
plots = [iv_smile_svma, iv_bs, iv_sabr]

for i, (plot_data, color, label) in enumerate(zip(plots, colors, labels)):
    for j, t in enumerate(tau):
        plt.plot(strikes, plot_data[j], label=f'{label} τ={t:.1f}' if j == 0 else "", 
                 marker='o' if i == 0 else 's' if i == 1 else '^', linewidth=2, color=color, alpha=0.8)

plt.title('Implied Volatility Smile: SVMA vs Benchmarks')
plt.xlabel('Strike Price ($)')
plt.ylabel('Implied Volatility')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svma_smile_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("Fixed smile curve saved as 'svma_smile_fixed.png'")
print("SABR curve now visible with equity-like skew!")
