
import numpy as np
import shap


def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    positive_idx = list(model.classes_).index(1)

    if isinstance(shap_values, list):
        shap_values = shap_values[positive_idx]
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, positive_idx]

    return shap_values
