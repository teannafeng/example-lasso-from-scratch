# %%
from src.lasso import LASSO
from src.plot import plot_lasso_ols
from src.print import print_lasso
from src.utils import run_ols_reg

# %%
# load data
from data.crime import load_crime_data
X, y, true_betas = load_crime_data()

# %%
# simulate data
# from src.simulate import simulate_xy_data
# X, y, true_betas = simulate_xy_data(n=100, p=15)

# %%
# fit model
mod = LASSO()
lasso_res = mod.run(y, X, cv_folds=10)

# %%
# print lasso results

print("\n>>> Print lasso results.")
print_lasso(lasso_res, true_betas, digits=2)

# %%
# run ols regression with active predictors selected by lasso

print("\n>>> Run ols regression with active predictors selected by lasso.")
print(run_ols_reg(y, X, subset=[0,1,2], digits=2))

# %%
# plot estimated betas and selected lambda

print("\n>>> Plot estimated betas and selected lambda.")
plot_lasso_ols(lasso_res, y, X, threshold=0.3)

# %%
# run ols regression with all predictors

print("\n>>> Run ols regression with all predictors.")
print(run_ols_reg(y, X, digits=2))

# %% 
# run with lambda = 0 + no cv --> ols regression with all predictors

print("\n>>> Run with lambda = 0 + no cv --> ols regression with all predictors.")
res_ = mod.run(y, X, lams=[0], cv_folds=0, max_iter=100)
print_lasso(res_, true_betas, digits=2)

# %%
