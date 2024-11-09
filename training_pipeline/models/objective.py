
import pandas as pd
from dataclasses import dataclass
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import optuna as opt
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier

@dataclass
class SVCObjective:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    data_type: dict[str, list[str]]

    def _get_params(self):
        pass
    
    def create_pipeline(self)->Pipeline:

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


    def __call__(self, trial:opt.trial.Trial) -> float:
        with mlflow.start_run(nested=True):
            pipeline = self.create_pipeline()
        # Define hyperparameters
            params = {
                "classifier__kernel": trial.suggest_categorical("classifier__kernel",choices=['linear', 'poly', 'rbf', 'sigmoid']),
                "classifier__C": trial.suggest_float("classifier__C", 0.001, 1.5, log=True)                
            }
            pipeline.set_params(**params)

            kf = StratifiedKFold(n_splits=5)
            cv = cross_val_score(pipeline,self.X_train,self.y_train,
                                 scoring="balanced_accuracy",cv=kf)
            cv_score = np.mean(cv)
            mlflow.log_params(params=params)
            mlflow.log_metric(key="score",value=cv_score)
            return cv_score
        