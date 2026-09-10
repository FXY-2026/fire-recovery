import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score


def train_rf_spatial(
    data,
    metric="MTR",
    features=None,
    block_size=3.0,
    test_ratio=0.2,
    n_trials=30,
    random_state=42
):
    """
    Predictors:
        RdNBR, Pre_LAI, Precipitation, SWC, SRAD, TMMX, VPD
    """
    if features is None:
        features = ['RdNBR', 'Pre_LAI', 'Precipitation',
                    'SWC', 'SRAD', 'TMMX', 'VPD']
    data = data.dropna().copy()
    results = {}
    for forest_value, group_name in [(1, "Forest"), (0, "NonForest")]:
        d = data[data["IsForest"] == forest_value].copy()

        d["Block"] = (
            np.floor(d["Lat"] / block_size).astype(str) + "_" +
            np.floor(d["Lon"] / block_size).astype(str)
        )

        np.random.seed(random_state)
        blocks = d["Block"].unique()
        target_n = int(len(d) * test_ratio)

        for _ in range(50):
            test_blocks = []
            n = 0

            for block in np.random.permutation(blocks):
                if n >= target_n:
                    break
                test_blocks.append(block)
                n += (d["Block"] == block).sum()

            test = d["Block"].isin(test_blocks)
            train = ~test

            if (d.loc[train, "Label"].nunique() == 2 and
                    d.loc[test, "Label"].nunique() == 2):
                break

        X_train = d.loc[train, features].values
        y_train = d.loc[train, "Label"].values
        X_test = d.loc[test, features].values
        y_test = d.loc[test, "Label"].values
        groups = d.loc[train, "Block"].values

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "random_state": random_state,
                "n_jobs": -1
            }
            aucs = []
            cv = GroupKFold(n_splits=5)

            for tr, va in cv.split(X_train, y_train, groups=groups):
                model = RandomForestClassifier(**params).fit(X_train[tr], y_train[tr])
                prob = model.predict_proba(X_train[va])[:, 1]
                if len(np.unique(y_train[va])) > 1:
                    aucs.append(roc_auc_score(y_train[va], prob))
            return np.mean(aucs) if aucs else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({"random_state": random_state, "n_jobs": -1})

        rf = RandomForestClassifier(**best_params).fit(X_train, y_train)

        prob = rf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        tag = f"{metric}_{group_name}"
        joblib.dump(rf, f"rf_model_{tag}.joblib")
        pd.DataFrame(X_test, columns=features).to_csv(
            f"X_test_{tag}.csv", index=False
        )
        results[group_name] = {
            "model": rf,
            "auc": roc_auc_score(y_test, prob),
            "acc": accuracy_score(y_test, pred),
            "best_params": best_params
        }
    return results