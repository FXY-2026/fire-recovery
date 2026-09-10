
import pandas as pd
import shap
import joblib


def compute_shap_values(
    metric: str = "MTR",
    groups: list = None,
    model_prefix: str = "rf_model_",
    xtest_prefix: str = "X_test_",
    output_prefix: str = "SHAP_values_"
):
    """
    Parameters
    ----------
    metric : str
        "MTR" or "RR"
    groups : list
        Default ["Forest", "NonForest"]
    """
    if groups is None:
        groups = ["Forest", "NonForest"]

    results = {}

    for group_name in groups:
        tag = f"{metric}_{group_name}"

        rf = joblib.load(f"{model_prefix}{tag}.joblib")
        X_test = pd.read_csv(f"{xtest_prefix}{tag}.csv")

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_test)
        sv = sv[:, :, 1]           

        shap_df = pd.DataFrame(sv, columns=X_test.columns)
        shap_df.to_csv(f"{output_prefix}{tag}.csv", index=False)

        results[group_name] = {
            "shap_values": shap_df,
            "feature_names": list(X_test.columns)
        }

    return results