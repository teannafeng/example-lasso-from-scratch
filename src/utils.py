from typing import Optional, List
import numpy as np
import pandas as pd
import typer

def run_ols_reg(
        y: np.ndarray, 
        X: np.ndarray, 
        subset: Optional[List[int]] = None, 
        digits: int = 3
    ):
    # get rid of the intercept
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    
    # subset predictors
    if subset is not None:
        if len(subset) == 0:
            typer.secho("Empty subset; skipp refit.", fg=typer.colors.RED)
            return pd.DataFrame(columns=["Est.", "SE", "Z"])
        
        X_c = X_c[:, subset]
    
    n, p = X_c.shape

    # check rank
    r = np.linalg.matrix_rank(X_c)
    if r < min(n, p):
        typer.secho(f"X is not full rank (rank={r}); skip refit.", fg=typer.colors.RED)
        return pd.DataFrame(columns=["Est.", "SE", "Z"])
    
    # ols estimates
    betas, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)

    if n <= p:
        typer.secho(f"No df for SE/Z (df={n-p}, n={n}, p={p}).", fg=typer.colors.YELLOW)
        idx = [f"b{j+1}" for j in (subset if subset is not None else range(X.shape[1]))]
        out = pd.DataFrame({"Est.": betas, "SE": np.nan, "Z": np.nan}, index=idx)
        return out.round(digits) if digits is not None else out
    
    # residuals and sigma^2
    resid = y_c - X_c @ betas
    sigma2 = resid @ resid / (n - p)
    
    # covariance matrix
    XtX_inv = np.linalg.inv(X_c.T @ X_c)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    
    # z-scores
    z_scores = betas / se
    
    # create table
    if subset is None:
        pred_idx = [f"b{j+1}" for j in range(X.shape[1])]
    else:
        pred_idx = [f"b{j+1}" for j in subset]
    
    out = pd.DataFrame({
        "Est.": betas,
        "SE": se,
        "Z": z_scores
    }, index=pred_idx)

    if digits is not None:
        out = out.round(digits)
    
    return out
