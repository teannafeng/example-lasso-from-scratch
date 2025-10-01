This repository implements a minimal textbook **LASSO (Least Absolute Shrinkage and Selection Operator)** using coordinate descent, with support for:

- Cross-validation for λ selection (based on MSE)
- Coefficient path across a sequence of λ values
- Optional OLS regression refit on active predictors selected by LASSO

**Notes:**

- Predictors are standardized and the dependent variable is centered internally
- No warm starts and no strong rules
- Standard error estimation is not yet implemented (could be added via bootstrap or debiasing)
- Written only to clarify the algorithm and to bridge concepts such as the soft-thresholding operator and coordinate descent

## Reference

Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical learning with sparsity: The lasso and generalizations*. Chapman and Hall/CRC. [https://doi.org/10.1201/b18401](https://doi.org/10.1201/b18401)

## Installation

### Clone the repo

```bash
git clone https://github.com/teannafeng/example-lasso-from-scratch.git
cd example-lasso-from-scratch
```

### Create virtual environment

```bash
python3 -m venv .venv       # if on Mac/Linux
python -m venv .venv        # if on Windows
```

### Activate it

```bash
source .venv/bin/activate   # if on Mac/Linux
.venv\Scripts\activate      # if on Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the demo in terminal:

```bash
python -m example.demo
```

Or open `./example/demo.py` in VS code and run the `# %%` cells.

## Folder structure

```text
example-lasso-from-scratch/
│
├── src/                # lasso implementation and helpers
│   ├── __init__.py
│   ├── lasso.py        
│   ├── plot.py
│   ├── print.py
│   └── simulate.py
│
├── example/
│   └── demo.py         # example
│
├── .vscode/            # vs code settings (if use vs code)
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt    # package dependencies
```
