import argparse, json, warnings
from pathlib import Path

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, optuna, pandas as pd, shap
from lightgbm import LGBMClassifier
from sklearn.datasets import fetch_openml, make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED, OUT = 42, Path("outputs")
OUT.mkdir(exist_ok=True)


def load_data(real):
    if real:
        d = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
        return d.data.astype(float), d.target.astype(int)
    X, y = make_classification(n_samples=60_000, n_features=30, n_informative=12,
                               n_redundant=6, weights=[0.99, 0.01], flip_y=0.002,
                               class_sep=1.0, random_state=SEED)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(30)]), pd.Series(y)


def tune_xgb(X, y, n_trials, spw):
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def objective(t):
        m = XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 150, 500),
            max_depth=t.suggest_int("max_depth", 3, 8),
            learning_rate=t.suggest_float("lr", 0.02, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample", 0.5, 1.0),
            min_child_weight=t.suggest_int("mcw", 1, 10),
            scale_pos_weight=spw, tree_method="hist", n_jobs=-1,
            eval_metric="aucpr", random_state=SEED)
        return cross_val_score(m, X, y, cv=cv, scoring="average_precision").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    return study.best_params


def best_threshold(y, p, cost_fn=50, cost_fp=1):
    ths = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in ths:
        tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    return float(ths[int(np.argmin(costs))])


def main(a):
    X, y = load_data(a.real)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"Rows: {len(X):,} | fraud rate: {y.mean():.3%}")

    bp = tune_xgb(Xtr, ytr, a.trials, spw)
    xgb = XGBClassifier(n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
                        learning_rate=bp["lr"], subsample=bp["subsample"],
                        colsample_bytree=bp["colsample"], min_child_weight=bp["mcw"],
                        scale_pos_weight=spw, tree_method="hist", n_jobs=-1, random_state=SEED)
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          scale_pos_weight=spw, random_state=SEED, verbose=-1)
    model = StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, stack_method="predict_proba", n_jobs=1)
    model.fit(Xtr, ytr)

    thr = best_threshold(yval, model.predict_proba(Xval)[:, 1])
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= thr).astype(int)
    metrics = {"roc_auc": roc_auc_score(yte, p),
               "pr_auc": average_precision_score(yte, p), "threshold": thr}
    print(json.dumps(metrics, indent=2))
    print(classification_report(yte, pred, digits=3))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pr, rc, _ = precision_recall_curve(yte, p)
    plt.figure(figsize=(6, 4.5)); plt.plot(rc, pr, lw=2)
    plt.axhline(yte.mean(), ls="--", c="gray", label="Baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP = {metrics['pr_auc']:.3f})"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "pr_curve.png", dpi=150); plt.close()

    xgb_fit = model.named_estimators_["xgb"]
    sample = Xte.sample(min(2000, len(Xte)), random_state=SEED)
    sv = shap.TreeExplainer(xgb_fit).shap_values(sample)
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=150); plt.close()

    joblib.dump({"model": model, "threshold": thr, "features": list(X.columns)},
                OUT / "model.joblib")
    print("Saved model + plots to ./outputs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--trials", type=int, default=15)
    main(ap.parse_args())