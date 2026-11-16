import argparse
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold


FEATURE_COLS = [
    "RdNBR", "Pre_LAI", "precipitation", "SWC",
    "SRAD", "TMMX", "VPD"
]
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 30
TEST_RATIO = 0.20



def prepare_samples(data, threshold=2):
    d = data.copy()
    d = d.loc[d["MTR"].abs() >= threshold].copy()
    d["Label"] = (d["MTR"] >= threshold).astype(np.int8)
    return d.reset_index(drop=True)


def add_spatial_blocks(data, block_size):
    d = data.copy()
    lat_block = np.floor(d["Lat"].to_numpy() / block_size).astype(int)
    lon_block = np.floor(d["Lon"].to_numpy() / block_size).astype(int)
    d["Block_ID"] = (
        pd.Series(lat_block, index=d.index).astype(str)
        + "_"
        + pd.Series(lon_block, index=d.index).astype(str)
    )
    return d


def spatial_train_test_split(
    data, block_size=3.0, test_ratio=TEST_RATIO,
    random_state=RANDOM_STATE, max_attempts=200
):
    d = add_spatial_blocks(data, block_size)
    blocks = d["Block_ID"].unique()
    target_n = int(len(d) * test_ratio)
    rng = np.random.RandomState(random_state)

    for _ in range(max_attempts):
        test_blocks, n_test = [], 0
        for block in rng.permutation(blocks):
            if n_test >= target_n:
                break
            test_blocks.append(block)
            n_test += (d["Block_ID"] == block).sum()

        test_mask = d["Block_ID"].isin(test_blocks)
        train_df = d.loc[~test_mask].copy()
        test_df = d.loc[test_mask].copy()

        if train_df["Label"].nunique() == 2 and test_df["Label"].nunique() == 2:
            return train_df, test_df

    raise RuntimeError("Unable to create a valid spatial train/test split.")


def make_spatial_cv_folds(
    X, y, groups, n_splits=N_FOLDS,
    random_state=RANDOM_STATE, max_attempts=200
):
    for seed in range(random_state, random_state + max_attempts):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        folds = list(cv.split(X, y, groups=groups))
        if all(
            len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2
            for tr, va in folds
        ):
            return folds

    raise RuntimeError("Unable to construct valid spatial CV folds.")


def train_rf_spatial(
    data, ecosystem, output_dir, features=None, 
    block_size=3.0, test_ratio=TEST_RATIO, n_trials=N_TRIALS,
    n_splits=N_FOLDS, random_state=RANDOM_STATE
):
    features = features or FEATURE_COLS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = spatial_train_test_split(
        data, block_size, test_ratio, random_state
    )

    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df["Label"].to_numpy(dtype=np.int8)
    groups = train_df["Block_ID"].to_numpy()
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df["Label"].to_numpy(dtype=np.int8)

    folds = make_spatial_cv_folds(
        X_train, y_train, groups, n_splits, random_state
    )

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
        for tr, va in folds:
            model = RandomForestClassifier(**params)
            model.fit(X_train[tr], y_train[tr])
            prob = model.predict_proba(X_train[va])[:, 1]
            aucs.append(roc_auc_score(y_train[va], prob))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update({"random_state": random_state, "n_jobs": -1})

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int8)

    metrics = {
        "CV_AUC": study.best_value,
        "Test_AUC": roc_auc_score(y_test, prob),
        "Test_Accuracy": accuracy_score(y_test, pred),
        "Test_Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "Test_Precision": precision_score(y_test, pred, zero_division=0),
        "Test_Recall": recall_score(y_test, pred, zero_division=0)
    }

    joblib.dump(model, output_dir / f"RF_model_{ecosystem}.joblib")


    return {
        "model": model,
        "study": study,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
        "best_params": best_params
    }


def run_random_forest_analysis(
    data, output_dir, threshold=2, block_size=3.0, features=None,
    test_ratio=TEST_RATIO, n_trials=N_TRIALS, random_state=RANDOM_STATE
):
    features = features or FEATURE_COLS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = prepare_samples(data, threshold)
    results = {}

    for forest_value, ecosystem in [(1, "Forest"), (0, "NonForest")]:
        subset = samples.loc[samples["IsForest"] == forest_value].copy()

        if subset["Label"].nunique() == 2:
            results[ecosystem] = train_rf_spatial(
                data=subset,
                ecosystem=ecosystem,
                output_dir=output_dir,
                features=features,
                threshold=threshold,
                block_size=block_size,
                test_ratio=test_ratio,
                n_trials=n_trials,
                random_state=random_state
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="RF_MTR_Models")
    parser.add_argument("--threshold", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--block-size", type=float, default=3.0, choices=[2.0, 3.0, 4.0])
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    run_random_forest_analysis(
        data=data,
        output_dir=args.output,
        threshold=args.threshold,
        block_size=args.block_size,
        n_trials=args.trials
    )


if __name__ == "__main__":
    main()
