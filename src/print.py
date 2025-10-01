import numpy as np

def _fmtf(x, fmt): 
    return format(float(x), fmt)

def print_lasso(lasso_res, true_betas=None, digits=3, use_unicode=False, show_bias=True):
    if isinstance(lasso_res, dict):
        betas = np.asarray(lasso_res["betas"])
        it    = lasso_res.get("iteration", None)
        obj   = lasso_res.get("obj", None)
        lam   = lasso_res.get("lambda", None)
    else:  # if is instance LassoResult
        betas = np.asarray(lasso_res.betas)
        it    = getattr(lasso_res, "iteration", None)
        obj   = getattr(lasso_res, "obj", None)
        lam   = getattr(lasso_res, "best_lambda", None)

    # header
    line = []
    if it is not None:
        line.append(f"Iter {it:4d}")
    if obj is not None:
        line.append(f"Obj: {obj:.6f}")
    if lam is not None:
        line.append(f"λ: {lam:.5f}")
    if line:
        print(" | ".join(line))
        print("")

    # numeric format
    fmt = f".{digits}f"

    # build string columns
    idx_col = [str(i+1) for i in range(len(betas))]
    est_col = [_fmtf(v, fmt) for v in betas]

    cols = []
    cols.append(("β" if use_unicode else "j", idx_col))
    if true_betas is not None:
        tb = [_fmtf(v, fmt) for v in np.asarray(true_betas)]
        cols.append(("True β" if use_unicode else "True b", tb))
    cols.append(("Est. β̂" if use_unicode else "Est. b", est_col))

    if true_betas is not None and show_bias:
        bias_col = [_fmtf(float(e) - float(t), fmt) for e, t in zip(betas, true_betas)]
        cols.append(("Bias", bias_col))

    # compute widths
    widths = [max(len(h), max(len(x) for x in col)) for h, col in cols]
    sep = "  "

    # header
    header = sep.join(f"{h:>{w}}" for (h, _), w in zip(cols, widths))
    print(header)
    print("-" * (sum(widths) + len(sep) * (len(cols) - 1)))

    # rows
    for i in range(len(idx_col)):
        row_vals = [col[i] for _, col in cols]
        print(sep.join(f"{v:>{w}}" for v, w in zip(row_vals, widths)))


