import matplotlib.pyplot as plt
import numpy as np

def plot_lasso_ols(res, y, X, threshold=0.3):
    # ols solution; center y, no intercept in regression
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    betas_ols = np.linalg.pinv(X_c) @ y_c
    beta_ols_norm = np.sum(np.abs(betas_ols))

    # compute relative L1 norms
    if isinstance(res, dict):
        betas_path = res["betas_path"]
    else:  # if is instance LassoResult
        betas_path = getattr(res, "betas_path", None)

    # append OLS as final point
    betas_path = np.vstack([betas_path, betas_ols])

    # compute relative L1 norms
    betas_ratio = np.sum(np.abs(betas_path), axis=1) / beta_ols_norm

    # get predictors survive at a larger lambda
    final_beta = betas_path[-1, :]
    survivors = np.where(np.abs(final_beta) > threshold)[0]

    plt.figure(figsize=(8, 6))
    for j in range(betas_path.shape[1]):
        label = fr"$\hat{{\beta}}_{{{j+1}}}$" if j in survivors else None
        plt.plot(betas_ratio, betas_path[:, j], label=label)

    plt.xlabel(r"$\|\hat{\beta}\|_{1} / \|\hat{\beta}_{OLS}\|_{1}$")
    plt.ylabel(r"Coefficient")
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=min(len(survivors), 8),
        frameon=False
    )
    plt.xlim(0, 1) 
    plt.tight_layout()
    plt.show()

