import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    w = np.random.randn(X.shape[-1])
    b = np.random.randn()

    n = X.shape[0]

    for _ in range(steps):
        z = X @ w + b
        out = _sigmoid(z)
    
        loss = (-1/n) * np.sum(y*np.log(out) + (1-y)*np.log(1-out))

        dz = out - y                  
        dw = (1/n) * (X.T @ dz)       
        db = (1/n) * np.sum(dz)       

        w -= lr * dw
        b -= lr * db

    return w, b