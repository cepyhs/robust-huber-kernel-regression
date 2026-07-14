# Save this module as robust_regressor.py
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel

class TikhonovHuberKernelRegression:
    def __init__(self, lambda_param=0.01, sigma=1.35, gamma=0.1):
        self.lambda_param = lambda_param  # Regularization parameter (\lambda)
        self.sigma = sigma                # Huber scaling scale parameter (\sigma)
        self.gamma = gamma                # RBF Mercer Kernel width parameter
        self.alpha = None                 # Optimization weight coefficients
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
