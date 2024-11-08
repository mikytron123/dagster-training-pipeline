import pandas as pd
from dagster import define_asset_job, asset, multi_asset, AssetOut, AssetIn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
import mlflow
import os

from models.utils import score_models

MLFLOW_HOST = os.getenv("MLFLOW_HOST",default="localhost")
MLFLOW_PORT = os.getenv("MLFLOW_PORT")

@multi_asset(outs={"df": AssetOut(key="df"), "data_type": AssetOut(key="data_type")})
def load_data() -> tuple[pd.DataFrame, dict]:
    """ """
    X, y = fetch_openml(data_id=179, return_X_y=True)
    df = pd.concat([X, y], axis="columns")
    print(df.shape)
    data_type_dict = dict(
        numeric_features=["fnlwgt"],
        categorical_features=[
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capitalgain",
            "capitalloss",
            "hoursperweek",
        ],
    )

    return df, data_type_dict


@multi_asset(
    ins={"df": AssetIn(key="df"), "data_type": AssetIn(key="data_type")},
    outs={"X": AssetOut(key="X"), "y": AssetOut(key="y")},
)
def preprocess(
    df: pd.DataFrame, data_type: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ """

    data_type_dict = data_type
    categorical_features = data_type_dict["categorical_features"]
    numeric_features = data_type_dict["numeric_features"]

    df = df.dropna(subset=categorical_features + numeric_features)
    df = df[df["workclass"] != "nan"]
    y = df[["class"]]
    X = df.drop(columns=["class"])[numeric_features + categorical_features]

    return X, y


@multi_asset(
    ins={"X": AssetIn(key="X"), "y": AssetIn(key="y")},
    outs={
        "X_train": AssetOut(key="X_train"),
        "X_test": AssetOut(key="X_test"),
        "y_train": AssetOut(key="y_train"),
        "y_test": AssetOut(key="y_test"),
    },
)
def train_test_splitter(
    X, y
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ """

    lbl = LabelBinarizer()
    y = lbl.fit_transform(y["class"]).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test


@asset(
    ins={
        "X_train": AssetIn(key="X_train"),
        "X_test": AssetIn(key="X_test"),
        "y_train": AssetIn(key="y_train"),
        "y_test": AssetIn(key="y_test"),
        "data_type": AssetIn(key="data_type"),
    }
)
def svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    data_type: dict[str, list[str]],
):
    mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

    mlflow.set_experiment(experiment_name="Training")
    mlflow.sklearn.autolog()

    data_type_dict = data_type
    categorical_features = data_type_dict["categorical_features"]
    numeric_features = data_type_dict["numeric_features"]

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train[y_train.columns[0]].to_numpy()
    elif isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        categories="auto", drop=None, handle_unknown="error"
    )
    mod = SVC(
        C=0.8,
        class_weight="balanced",
    )
    with mlflow.start_run(run_name="SVM"):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])

        dist = dict(classifier__C=uniform(loc=0.1, scale=1))

        mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5, verbose=3)
        mod_cv.fit(X_train, y_train)

        best_mod = mod_cv.best_estimator_

        mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score = (
            score_models(best_mod, X_test, y_test)
        )
        mlflow.log_metric(key="test_f1_score", value=mod_f1_score)
        mlflow.log_metric(key="test_precision_score", value=mod_precision_score)
        mlflow.log_metric(key="test_recall_score", value=mod_recall_score)
        mlflow.log_metric(key="test_balanced_accuracy_score", value=mod_accuracy_score)


@asset(ins={
        "X_train": AssetIn(key="X_train"),
        "X_test": AssetIn(key="X_test"),
        "y_train": AssetIn(key="y_train"),
        "y_test": AssetIn(key="y_test"),
        "data_type": AssetIn(key="data_type"),
    })
def decision_tree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    data_type: dict[str, list[str]],
):
    mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

    mlflow.set_experiment(experiment_name="Training")
    mlflow.sklearn.autolog()

    data_type_dict = data_type
    categorical_features = data_type_dict["categorical_features"]

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train[y_train.columns[0]].to_numpy()
    elif isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test[y_test.columns[0]].to_numpy()
    elif isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    categorical_transformer = OneHotEncoder(
        categories="auto", drop=None, handle_unknown="error"
    )
    mod = DecisionTreeClassifier(
        class_weight="balanced",
    )
    with mlflow.start_run(run_name="decision tree"):
        preprocessor = ColumnTransformer(
            transformers=[
                # ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
            force_int_remainder_cols=False,
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])

        dist = dict(
            classifier__min_samples_split=randint(low=2, high=15),
            classifier__min_samples_leaf=randint(low=1, high=15),
        )
        mlflow.log_params(
            {
                "classifier__min_samples_split": "randint(low=2, high=15)",
                "classifier__min_samples_leaf": "randint(low=1,high=15)",
            }
        )

        mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5, verbose=3)
        mod_cv.fit(X_train, y_train)

        best_mod = mod_cv.best_estimator_

        mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score = (
            score_models(best_mod, X_test, y_test)
        )
        mlflow.log_metric(key="test_f1_score", value=mod_f1_score)
        mlflow.log_metric(key="test_precision_score", value=mod_precision_score)
        mlflow.log_metric(key="test_recall_score", value=mod_recall_score)
        mlflow.log_metric(key="test_balanced_accuracy_score", value=mod_accuracy_score)


train_model_pipeline = define_asset_job(
    name="train_model_pipeline",
    selection=[
        load_data,
        preprocess,
        train_test_splitter,
        decision_tree,
        svm
    ],
)
