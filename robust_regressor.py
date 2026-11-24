import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel


class TikhonovHuberKernelRegression:
    """
    Tikhonov-regularized Huber regression in a Reproducing Kernel Hilbert Space (RKHS).
    Uses fixed heuristic hyperparameters (lambda_param, sigma, gamma) rather than the
    asymptotic sample-size schedule derived in Tong (2026).
    """

    def __init__(self, lambda_param=0.01, sigma=1.35, gamma=0.1):
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = None
        self.X_train = None

    def _huber_loss(self, residuals):
        abs_r = np.abs(residuals)
        return np.where(
            abs_r <= self.sigma,
            0.5 * (residuals ** 2),
            self.sigma * (abs_r - 0.5 * self.sigma),
        )

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
        result = minimize(objective_function, initial_alpha, method="BFGS")
        self.alpha = result.x
        return self

    def predict(self, X_new):
        K_new = rbf_kernel(X_new, self.X_train, gamma=self.gamma)
        return K_new.dot(self.alpha)


if __name__ == "__main__":
    print("Fetching BTC-USD data from Yahoo Finance...")
    data = yf.download("BTC-USD", start="2025-01-01", progress=False)
    y = data["Close"].values.flatten()
    X = np.arange(len(y)).reshape(-1, 1)

    # Normalize features and target
    X_scaled = (X - X.min()) / (X.max() - X.min()) * 5.0
    y_scaled = (y - y.min()) / (y.max() - y.min()) * 10.0

    # Inject two synthetic extreme outliers at 25% and 75% of the series
    idx_1 = int(len(y) * 0.25)
    idx_2 = int(len(y) * 0.75)
    y_anomalous = y_scaled.copy()
    y_anomalous[idx_1] -= 5.0
    y_anomalous[idx_2] += 5.0

    # Hyperparameters
    lam = 0.008
    sig = 0.5
    gam = 0.3

    # 1. Fit Tikhonov-Huber RKHS model
    print(f"Fitting Huber-Tikhonov-RKHS model on {len(y)} points...")
    huber_model = TikhonovHuberKernelRegression(lambda_param=lam, sigma=sig, gamma=gam)
    huber_model.fit(X_scaled, y_anomalous)

    # 2. Fit Kernel Ridge Regression (same RKHS & regularization weight, L2 loss)
    krr_model = KernelRidge(alpha=lam * len(y), kernel="rbf", gamma=gam)
    krr_model.fit(X_scaled, y_anomalous)

    # 3. Fit Ordinary Least Squares (linear L2 baseline)
    ols_model = LinearRegression()
    ols_model.fit(X_scaled, y_anomalous)

    # Evaluate sensitivity at the injected anomaly coordinates
    X_grid = np.linspace(0, 5, 300).reshape(-1, 1)
    y_pred_huber = huber_model.predict(X_grid)
    y_pred_krr = krr_model.predict(X_grid)
    y_pred_ols = ols_model.predict(X_grid)

    print("\n--- Prediction Values at Injected Anomaly Indices ---")
    print(f"Original clean values : {np.round(y_scaled[[idx_1, idx_2]], 3)}")
    print(f"Corrupted targets     : {np.round(y_anomalous[[idx_1, idx_2]], 3)}")
    print(f"Huber-RKHS fit        : {np.round(huber_model.predict(X_scaled[[idx_1, idx_2]]), 3)}")
    print(f"Kernel Ridge (L2) fit : {np.round(krr_model.predict(X_scaled[[idx_1, idx_2]]), 3)}")
    print(f"Linear OLS fit        : {np.round(ols_model.predict(X_scaled[[idx_1, idx_2]]), 3)}\n")

    # Save comparison plot
    plt.figure(figsize=(12, 6))
    plt.scatter(
        X_scaled,
        y_anomalous,
        color="red",
        s=15,
        alpha=0.45,
        label="BTC-USD normalized price (with 2 injected anomalies)",
    )
    plt.plot(X_grid, y_pred_huber, color="blue", linewidth=2.5, label="Huber-Tikhonov-RKHS fit")
    plt.plot(X_grid, y_pred_krr, color="darkorange", linewidth=2, linestyle="-.", label="Kernel Ridge (L2-RKHS) fit")
    plt.plot(X_grid, y_pred_ols, color="green", linewidth=1.8, linestyle="--", label="Linear OLS baseline")
    plt.title("Robust Huber-Tikhonov RKHS vs. L2 Baselines on BTC-USD Data")
    plt.xlabel("Normalized timeline (Jan 2025 onward)")
    plt.ylabel("Normalized closing price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("robust_vs_OLS_regression.png", dpi=300)
    print("Saved comparison chart to robust_vs_OLS_regression.png")
