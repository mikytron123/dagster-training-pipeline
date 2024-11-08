import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def score_models(
    model:DecisionTreeClassifier | SVC , X: pd.DataFrame, y: np.ndarray
) -> tuple[float, float, float, float]:
    y_pred = model.predict(X)
    mod_f1_score = f1_score(y_true=y, y_pred=y_pred, average="micro")
    mod_precision_score = precision_score(y_true=y, y_pred=y_pred, average="micro")
    mod_recall_score = recall_score(y_true=y, y_pred=y_pred, average="micro")
    mod_accuracy_score = balanced_accuracy_score(
        y_true=y,
        y_pred=y_pred,
    )
    return mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score
