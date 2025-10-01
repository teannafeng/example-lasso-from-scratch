import numpy as np

def simulate_xy_data(n=200, p=20, true_beta=None, sigma=0.5, seed=1010):
    np.random.seed(seed)
    X = np.random.randn(n, p)

    if true_beta is None:
        beta_base = [1.2, -1.5, 2.3]
        true_beta = np.array(beta_base + [0]*(p - len(beta_base)))
    else:
        true_beta = np.array(true_beta)
        if len(true_beta) < p:
            true_beta = np.concatenate([true_beta, np.zeros(p - len(true_beta))])
        elif len(true_beta) > p:
            raise ValueError("No. true betas cannot exceed the specified p")

    y = X @ true_beta + np.random.randn(n) * sigma
    return X, y, true_beta