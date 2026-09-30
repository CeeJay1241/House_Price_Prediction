"""Compare models on the full California dataset with random and spatial CV.

Random K-fold lets nearby houses land in both train and test folds (location
leakage). Spatial CV clusters houses by lat/long and holds out whole regions,
which is a more honest estimate of performance on unseen areas.

Usage:
  python -m src.evaluate
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_california
from src.features import build_preprocessor, df_to_Xy, get_numeric_features

ROOT = Path(__file__).resolve().parents[1]


def _models(seed: int) -> dict:
    return {
        "ridge": Ridge(),
        "rf": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "gboost": HistGradientBoostingRegressor(max_iter=300, random_state=seed),
    }


def compare_random_vs_spatial(sample_n: int | None = None, n_regions: int = 20, seed: int = 42) -> dict:
    X, y = df_to_Xy(load_california(sample_n=sample_n))
    feats = get_numeric_features(X)
    regions = KMeans(n_regions, random_state=seed, n_init=10).fit_predict(X[["Latitude", "Longitude"]])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    def rmse_cv(model, cv, groups=None):
        pipe = Pipeline([("pre", build_preprocessor(feats)), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="neg_mean_squared_error")
        return float(np.sqrt(-scores).mean())

    results = {}
    for name, model in _models(seed).items():
        pipe = Pipeline([("pre", build_preprocessor(feats)), ("model", model)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "test_rmse": float(mean_squared_error(y_te, pred) ** 0.5),
            "test_mae": float(mean_absolute_error(y_te, pred)),
            "test_r2": float(r2_score(y_te, pred)),
            "random_cv_rmse": rmse_cv(model, KFold(5, shuffle=True, random_state=seed)),
            "spatial_cv_rmse": rmse_cv(model, GroupKFold(5), groups=regions),
        }
    return results


if __name__ == "__main__":
    res = compare_random_vs_spatial()
    print(f"{'model':8} {'R2':>6} {'RMSE':>6} {'randomCV':>9} {'spatialCV':>10}")
    for n, r in res.items():
        print(f"{n:8} {r['test_r2']:6.3f} {r['test_rmse']:6.3f} {r['random_cv_rmse']:9.3f} {r['spatial_cv_rmse']:10.3f}")
    out = ROOT / "configs" / "eval_results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Wrote", out)
