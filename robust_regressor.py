import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf


class TikhonovHuberKernelRegression:
    """
    Tikhonov-regularized Huber regression in RKHS.

    Note: sigma and lambda_param are set manually here. Tong (2026,
    Applied and Computational Harmonic Analysis) shows theoretical
    generalization guarantees hold when these are jointly tuned according
    to explicit sample-size-dependent formulas (see Theorem 1 of that
    paper). The fixed values below are a practical heuristic, not a
    theoretically guaranteed configuration.
    """

    def __init__(self, lambda_param=0.01, sigma=1.35, gamma=0.1):
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = None
        self.X_train = None

    def _huber_loss(self, w):
        abs_w = np.abs(w)
        return np.where(abs_w <= self.sigma, 0.5 * (w**2), self.sigma * (abs_w - 0.5 * self.sigma))

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        K = rbf_kernel(X, X, gamma=self.gamma)

        def objective_function(alpha):
            predictions = K.dot(alpha)
            residuals = y - predictions
            loss_term = np.mean(self._huber_loss(residuals))
            reg_term = self.lambda_param * alpha.dot(K).dot(alpha)
            return loss_term + reg_term

        initial_alpha = np.zeros(n_samples)
        result = minimize(objective_function, initial_alpha, method='BFGS')
        self.alpha = result.x
        return self

    def predict(self, X_new):
        K_new = rbf_kernel(X_new, self.X_train, gamma=self.gamma)
        return K_new.dot(self.alpha)


# 1. DOWNLOAD REAL-WORLD BITCOIN DATA
print("Fetching BTC-USD data from Yahoo Finance...")
data = yf.download("BTC-USD", start="2025-01-01")
y = data['Close'].values.flatten()
X = np.arange(len(y)).reshape(-1, 1)

# Normalize
X_scaled = (X - X.min()) / (X.max() - X.min()) * 5
y_scaled = (y - y.min()) / (y.max() - y.min()) * 10

# Inject synthetic anomalies at 25% / 75% marks (clearly labeled as synthetic —
# these are large, controlled injections used to visualize outlier response,
# not real flash-crash events)
idx_1 = int(len(y) * 0.25)
idx_2 = int(len(y) * 0.75)
y_scaled_anomalous = y_scaled.copy()
y_scaled_anomalous[idx_1] -= 5.0
y_scaled_anomalous[idx_2] += 5.0

# 2. FIT ROBUST HUBER-KERNEL MODEL
print(f"Fitting Huber-Tikhonov model on {len(y)} points...")
huber_model = TikhonovHuberKernelRegression(lambda_param=0.008, sigma=0.5, gamma=0.3)
huber_model.fit(X_scaled, y_scaled_anomalous)

# 3. FIT AN ORDINARY LEAST SQUARES BASELINE FOR COMPARISON
# (included so the robustness claim is visually and numerically checkable,
# rather than asserted without a baseline)
print("Fitting OLS baseline for comparison...")
ols_model = LinearRegression()
ols_model.fit(X_scaled, y_scaled_anomalous)

# 4. GENERATE PREDICTIONS
X_grid = np.linspace(0, 5, 300).reshape(-1, 1)
y_pred_huber = huber_model.predict(X_grid)
y_pred_ols = ols_model.predict(X_grid)

# Quantify how much each fit was pulled toward the injected anomalies
huber_pred_at_anomalies = huber_model.predict(X_scaled[[idx_1, idx_2]])
ols_pred_at_anomalies = ols_model.predict(X_scaled[[idx_1, idx_2]])
print(f"Huber fit value near anomaly points: {huber_pred_at_anomalies}")
print(f"OLS fit value near anomaly points:   {ols_pred_at_anomalies}")

# 5. VISUALIZE
plt.figure(figsize=(12, 6))
plt.scatter(X_scaled, y_scaled_anomalous, color='red', s=15, alpha=0.5,
            label='BTC price (with 2 synthetic injected anomalies)')
plt.plot(X_grid, y_pred_huber, color='blue', linewidth=2.5, label='Huber-Tikhonov-RKHS fit')
plt.plot(X_grid, y_pred_ols, color='green', linewidth=2, linestyle='--', label='OLS baseline fit')
plt.title('Robust vs. OLS Regression on BTC Price Data with Injected Anomalies')
plt.xlabel('Normalized timeline (Jan 2025 onward)')
plt.ylabel('Normalized price')
plt.legend()
plt.grid(True)
plt.savefig('huberloss_real_data.png', dpi=300)
print("Saved chart to huberloss_real_data.png")
plt.show()
