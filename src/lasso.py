import typer
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from typing import Optional, List, Union

class LassoResult:
    def __init__(
            self, 
            betas: np.ndarray, 
            intercept: float, 
            betas_path: np.ndarray, 
            lambdas: Union[List[float], np.ndarray], 
            cv_mse: np.ndarray, 
            best_lambda: float, 
            converged: Union[bool, int],
            iteration: int
        ):
        self.betas = betas
        self.intercept = intercept
        self.betas_path = betas_path
        self.lambdas = lambdas
        self.cv_mse = cv_mse
        self.best_lambda = best_lambda
        self.converged = converged
        self.iteration = iteration

    def summary(self):
        return {
            "best_lambda": self.best_lambda,
            "converged": self.converged,
            "iterations": self.iteration,
            "n_betas_selected": int((self.betas != 0).sum()),
            "betas": self.betas,
            "intercept": np.array([self.intercept]),
        }

class LASSO:
    def __init__(self):
        self.X_mea: Optional[np.ndarray] = None
        self.X_scale: Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None

    def _standardize(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            verbose: bool = True
        ):
        n, p = X.shape

        # center predictors
        self.X_mean = X.mean(axis=0)
        X_c = X - self.X_mean

        # scale each column s.t. (1/n) ||x_j||^2 = 1
        self.X_scale = np.sqrt((X_c ** 2).sum(axis=0) / n)
        zero_scale = self.X_scale == 0
        if np.any(zero_scale):
            typer.secho(
                f"Warning: {zero_scale.sum()} predictors with same values detected. "
                "These predictors carry no variance and will be ignored (coefficients forced to 0).", 
                fg=typer.colors.RED
            )
            self.X_scale[zero_scale] = 1.0   # avoid div by zero

        X_std = X_c / self.X_scale

        # center dv
        self.y_mean = y.mean()
        y_c = y - self.y_mean

        if verbose:
            typer.secho(
                "Standardized X (mean=0, variance=1) and centered y (mean=0).", 
                fg=typer.colors.YELLOW
            )

        return X_std, y_c
    
    def _rescale(self, betas_std: np.ndarray):
        betas = betas_std / self.X_scale
        intercept = self.y_mean - self.X_mean @ betas
        return betas, intercept
    
    def _gen_lambda_seq(
            self, 
            X_std: np.ndarray, 
            y_c: np.ndarray, 
            n_lams: int = 100, 
            lam_min_ratio: Optional[float] = None, 
            verbose: bool = True
        ):
        n, p = X_std.shape
        
        if lam_min_ratio is None:
            lam_min_ratio = 0.01 if n < p else 1e-4

        # compute lam_max
        lam_max = np.max(np.abs((X_std.T @ y_c) / n))
        lam_min = lam_min_ratio * lam_max

        # log-spaced lambda sequence
        lams = np.logspace(np.log10(lam_max), np.log10(lam_min), n_lams)

        if verbose:
            typer.secho(
                f"Auto-generated {n_lams} lambdas from {lam_max:.4f} down to {lam_min:.4f}.",
                fg=typer.colors.YELLOW
            )

        return lams
    
    def _soft_thresh_j(
            self, 
            n: int, 
            x_j: np.ndarray, 
            r_j: np.ndarray, 
            lam: float
        ):
        z = (x_j @ r_j) / n
        if z > lam:
            return (z - lam)
        elif z < -lam:
            return (z + lam)
        else:
            return 0.0
    
    def _check_convergence(
            self, 
            beta_old: np.ndarray, 
            beta_new: np.ndarray, 
            tol: float = 1e-6
        ):
        return np.max(np.abs(beta_new - beta_old)) < tol
    
    def _run(
            self, 
            y_c: np.ndarray, 
            X_std: np.ndarray, 
            lam: float, 
            init_betas: Optional[np.ndarray] = None,
            max_iter: int = 100, 
            tol: float = 1e-6, 
            verbose: bool = True
        ):
        n, p = X_std.shape
        converged = False
        betas_std = np.zeros(p) if init_betas is None else init_betas.copy()
        Xb = np.zeros(n)
        
        iterator = range(max_iter)
        if verbose:
            iterator = tqdm(iterator, desc=f"LASSO λ={lam:.4f}", leave=False)

        for iter in iterator:
            betas_old = betas_std.copy()

            for j in range(p):
                # compute partial residual for coordinate j
                r_j = y_c - (Xb - X_std[:, j] * betas_std[j])
                # update beta_j with soft-thresholding
                beta_j = self._soft_thresh_j(n, X_std[:, j], r_j, lam)
                # update fitted values
                Xb += X_std[:, j] * (beta_j - betas_std[j])
                # save updated beta_j
                betas_std[j] = beta_j
                
            if self._check_convergence(betas_old, betas_std, tol):
                converged = True
                break
        
        betas, intercept = self._rescale(betas_std)

        return {
            "converged": int(converged),
            "iteration": iter + 1, 
            "betas_std": betas_std,
            "betas"    : betas,
            "intercept": intercept,
            "lambda"   : lam,
        }
    
    def run(
        self, 
        y: np.ndarray, 
        X: np.ndarray, 
        lams: Optional[np.ndarray] = None, 
        n_lams: int = 100, 
        lam_min_ratio: float = 0.01, 
        cv_folds: int = 10, 
        cv_seed: int = 1010, 
        max_iter: int = 100, 
        tol: float = 1e-6, 
        verbose: bool = True
    ) -> LassoResult:
        
        n, p = X.shape
        
        if cv_folds < 2:
            typer.secho(
                "cv_folds < 2; no cross-validation will be used.", 
                fg=typer.colors.YELLOW
            )
            cv_folds = 0
        elif n < cv_folds:
            typer.secho(
                f"cv_folds={cv_folds} > sample size={n}; using cv_folds={n}.", 
                fg=typer.colors.YELLOW
            )
            cv_folds = n
        
        # standardize X and center y
        X_std, y_c = self._standardize(X, y, verbose)

        if lams is None:
            lams = self._gen_lambda_seq(X_std, y_c, n_lams=n_lams, lam_min_ratio=lam_min_ratio, verbose=verbose)

        typer.secho(f"Running LASSO with {len(lams)} lambdas.", fg=typer.colors.CYAN)

        avg_mse = []
        betas_path = []

        if cv_folds == 0:
            # no CV: juse the full dataset, evaluate mse
            typer.secho(
                "No cross-validation. Using full dataset MSE for lambda selection.", 
                fg=typer.colors.YELLOW
            )
            for lam in lams:
                run_res = self._run(y_c, X_std, lam, max_iter=max_iter, tol=tol, verbose=False)
                betas_path.append(run_res["betas"])  # original scale
                y_c_pred = X_std @ run_res["betas_std"]
                mse = np.mean((y_c - y_c_pred) ** 2)
                avg_mse.append(mse)
        else:
            # k-fold cv
            typer.secho(
                f"Performing {cv_folds}-fold cross-validation.", 
                fg=typer.colors.CYAN
            )
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
            for lam in lams:
                mse = []
                for train, val in kf.split(X_std):
                    run_res = self._run(y_c[train], X_std[train], lam, max_iter=max_iter, tol=tol, verbose=False)
                    y_c_pred = X_std[val] @ run_res["betas_std"]
                    mse.append(np.mean((y_c[val] - y_c_pred) ** 2))
                avg_mse.append(np.mean(mse))

                run_res = self._run(y_c, X_std, lam, max_iter=max_iter, tol=tol, verbose=False)
                betas_path.append(run_res["betas"])  # original scale

        # store history
        avg_mse = np.array(avg_mse)
        betas_path = np.array(betas_path)

        # find best lambda (lowest average mse)
        best_idx = np.argmin(avg_mse)
        best_lambda = lams[best_idx]

        if verbose:
            typer.secho(
                f"Best lambda selected: {best_lambda:.4f} (index {best_idx}).", 
                fg=typer.colors.GREEN
            )

        # final estimates at best lambda
        final_results = self._run(y_c, X_std, best_lambda, max_iter=max_iter, tol=tol, verbose=verbose)

        return LassoResult(
            betas=final_results["betas"],
            intercept=final_results["intercept"],
            betas_path=betas_path,
            lambdas=lams,
            cv_mse=avg_mse,
            best_lambda=best_lambda,
            converged=final_results["converged"],
            iteration=final_results["iteration"]
        )

