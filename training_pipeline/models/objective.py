from abc import ABC, abstractmethod
from typing import override, Mapping
from dataclasses import dataclass

import numpy as np
import optuna as opt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlflow


@dataclass
class Objective(ABC):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    data_type: dict[str, list[str]]

    @abstractmethod
    def create_pipeline(self) -> Pipeline:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, trial: opt.trial.Trial) -> float:
        raise NotImplementedError

    def train_model(self, params: Mapping[str, int | str | float]) -> float:
        pipeline = self.create_pipeline()
        pipeline.set_params(**params)

        kf = StratifiedKFold(n_splits=5)
        cv:np.typing.NDArray = cross_val_score(
            pipeline, self.X_train, self.y_train, scoring="balanced_accuracy", cv=kf
        )
        cv_score = float(np.mean(cv))
        # mlflow.log_params(params=params)
        _ = mlflow.log_metric(key="score", value=float(cv_score))
        return cv_score


@dataclass
class SVCObjective(Objective):
    @override
    def create_pipeline(self) -> Pipeline:
        categorical_features = self.data_type["categorical_features"]
        numeric_features = self.data_type["numeric_features"]

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            categories="auto", drop=None, handle_unknown="error"
        )
        mod = SVC(
            class_weight="balanced",
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])
        return clf
    @override
    def __call__(self, trial: opt.trial.Trial) -> float:
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                "classifier__kernel": trial.suggest_categorical(
                    "classifier__kernel", choices=["linear", "poly", "rbf", "sigmoid"]
                ),
                "classifier__C": trial.suggest_float(
                    "classifier__C", 0.001, 1.5, log=True
                ),
            }
            return self.train_model(params)


@dataclass
class DecisionTreeObjective(Objective):
    def create_pipeline(self) -> Pipeline:
        categorical_features = self.data_type["categorical_features"]
        numeric_features = self.data_type["numeric_features"]

        categorical_transformer = OneHotEncoder(
            categories="auto", drop=None, handle_unknown="error"
        )
        mod = DecisionTreeClassifier(
            class_weight="balanced",
        )

        preprocessor = ColumnTransformer(
            transformers=[
                # ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough"
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])
        return clf
    @override
    def __call__(self, trial: opt.trial.Trial) -> float:
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                "classifier__max_depth": trial.suggest_int(
                    "classifier__max_depth", 3, 15
                ),
                "classifier__min_samples_split": trial.suggest_int(
                    "classifier__min_samples_split", 2, 20
                ),
                "classifier__min_samples_leaf": trial.suggest_int(
                    "classifier__min_samples_leaf", 1, 20
                ),
                # "classifier__max_features": trial.suggest_categorical("classifier__max_features",["auto"])
            }
            return self.train_model(params)
