import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class BaselineModels:
    def __init__(self):
        self.ridge = None
        self.scaler = StandardScaler()
        
    def train_ridge(self, X_train, y_train, alpha=1.0):
        """Trains a simple Ridge regression baseline."""
        # Scale data for linear model
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(X_scaled, y_train)
        
    def predict(self, X):
        """Returns predictions in a dictionary."""
        if self.ridge is None:
            raise ValueError("Ridge model not trained yet.")
            
        X_scaled = self.scaler.transform(X)
        preds = self.ridge.predict(X_scaled)
        
        return {'ridge': preds}
