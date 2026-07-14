import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# 1. DEFINE THE MODEL CLASS (FROM THE RESEARCH PAPER)
class TikhonovHuberKernelRegression:
    def __init__(self, lambda_param=0.01, sigma=1.35, gamma=0.1):
        self.lambda_param = lambda_param  # \lambda (Tikhonov capacity penalty)
        self.sigma = sigma                # \sigma (Huber loss threshold parameter)
        self.gamma = gamma                # Width parameter for RBF Mercer Kernel
        self.alpha = None                 # Optimization weight coefficients to be learned
        self.X_train = None
        
    def _huber_loss(self, w):
        """Calculates Huber Loss based on the error value"""
        abs_w = np.abs(w)
        return np.where(abs_w <= self.sigma, 0.5 * (w**2), self.sigma * (abs_w - 0.5 * self.sigma))

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        
        # Compute the Kernel Matrix (K) using RBF Kernel
        K = rbf_kernel(X, X, gamma=self.gamma)
        
        # Define the objective function to minimize (Equation 2.2 from the paper)
        def objective_function(alpha):
            predictions = K.dot(alpha)
            residuals = y - predictions
            loss_term = np.mean(self._huber_loss(residuals))
            reg_term = self.lambda_param * alpha.dot(K).dot(alpha)
            return loss_term + reg_term

        # Run optimization solver to find the best alpha coefficients
        initial_alpha = np.zeros(n_samples)
        result = minimize(objective_function, initial_alpha, method='BFGS')
        
        self.alpha = result.x
        return self

    def predict(self, X_new):
        # Compute kernel between new data and training data
        K_new = rbf_kernel(X_new, self.X_train, gamma=self.gamma)
        return K_new.dot(self.alpha)

# 2. GENERATE SYNTHETIC TRAINING DATA WITH SEVERE OUTLIERS
np.random.seed(42)
X = np.random.rand(40, 1) * 5  # 40 random data points between 0 and 5
# True underlying function: y = 2x + 1 + Gaussian noise
y = 2 * X.squeeze() + 1 + np.random.normal(0, 0.3, 40)

# Inject intentional extreme anomalies (heavy-tailed outliers)
y[5] += 15.0  
y[25] -= 15.0

# 3. TRAIN THE ROBUST REGRESSION MODEL
model = TikhonovHuberKernelRegression(lambda_param=0.001, sigma=1.0, gamma=0.1)
model.fit(X, y)

# 4. GENERATE PREDICTIONS FOR PLOTTING
X_grid = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = model.predict(X_grid)

# 5. VISUALIZE THE ROBUST PERFORMANCE WITH AN ENGLISH CHART
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Corrupted Training Data (With Outliers)')
plt.plot(X_grid, y_pred, color='blue', linewidth=2, label='Robust Model Prediction (Huber-Tikhonov)')
plt.title('Robust Regression Performance Under Severe Outlier Injections')
plt.xlabel('X (Input Variable)')
plt.ylabel('y (Output Response)')
plt.legend()
plt.grid(True)

# Force the plot window to display and hold in local environments
plt.show()
