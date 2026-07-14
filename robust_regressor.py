import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import yfinance as yf

# 1. DEFINE THE MODEL CLASS (FROM THE RESEARCH PAPER)
class TikhonovHuberKernelRegression:
    def __init__(self, lambda_param=0.01, sigma=1.35, gamma=0.1):
        self.lambda_param = lambda_param  
        self.sigma = sigma                
        self.gamma = gamma                
        self.alpha = None                 
        self.X_train = None
        
    def _huber_loss(self, w):
        """Calculates Huber Loss based on the error value"""
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

# 2. DOWNLOAD REAL-WORLD VOLATILE FINANCIAL DATA (JAN 2025 - PRESENT)
print("Fetching real-world data from Yahoo Finance...")
# Removing 'end' parameter automatically fetches data up to the current date in 2026
data = yf.download("BTC-USD", start="2025-01-01")

# Extract close prices
y = data['Close'].values.flatten()
X = np.arange(len(y)).reshape(-1, 1)

# Normalization (Scale to 0-5 window to optimize kernel performance)
X_scaled = (X - X.min()) / (X.max() - X.min()) * 5
y_scaled = (y - y.min()) / (y.max() - y.min()) * 10

# Inject artificial flash crash/anomalies into the expanded timeline
# We place them at around 25% and 75% marks of the dataset dynamically
idx_1 = int(len(y) * 0.25)
idx_2 = int(len(y) * 0.75)
y_scaled[idx_1] -= 5.0  
y_scaled[idx_2] += 5.0

# 3. TRAIN THE ROBUST REGRESSION MODEL
print(f"Training the robust Huber-Tikhonov model on {len(y)} data points...")
# Adjusted hyperparameters slightly to balance the larger dataset size
model = TikhonovHuberKernelRegression(lambda_param=0.008, sigma=0.5, gamma=0.3)
model.fit(X_scaled, y_scaled)

# 4. GENERATE PREDICTIONS FOR PLOTTING
X_grid = np.linspace(0, 5, 300).reshape(-1, 1)
y_pred = model.predict(X_grid)

# 5. VISUALIZE PERFORMANCE UP TO THE CURRENT PERIOD
plt.figure(figsize=(12, 6))
plt.scatter(X_scaled, y_scaled, color='red', s=15, alpha=0.6, label='Actual Bitcoin Price (With Anomaly Injections)')
plt.plot(X_grid, y_pred, color='blue', linewidth=2.5, label='Robust Model Trend Line')
plt.title('Robust Kernel Regression on Bitcoin Volatility (Jan 2025 - Present)')
plt.xlabel('Normalized Timeline (January 2025 onwards)')
plt.ylabel('Normalized Target Price')
plt.legend()
plt.grid(True)

# Save the comprehensive chart for your portfolio
plt.savefig('huberloss_real_data.png', dpi=300)
print("Success! Updated chart saved as huberloss_real_data.png")
plt.show()
