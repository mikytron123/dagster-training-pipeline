import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def score_models(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.typing.NDArray,  # pyright: ignore[reportMissingTypeArgument,reportUnknownMemberType,reportUnknownParameterType]
) -> tuple[float, float, float, float]:
    y_pred: np.typing.NDArray = model.predict(
        X
    )  # pyright: ignore[reportUnknownMemberType,reportMissingTypeArgument,reportUnknownVariableType]
    mod_f1_score = f1_score(y_true=y, y_pred=y_pred, average="micro")
    mod_precision_score = precision_score(y_true=y, y_pred=y_pred, average="micro")
    mod_recall_score = recall_score(y_true=y, y_pred=y_pred, average="micro")
    mod_accuracy_score = balanced_accuracy_score(
        y_true=y,
        y_pred=y_pred,
    )
    return mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score
