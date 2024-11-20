import pandas as pd
from dagster import define_asset_job, asset, multi_asset, AssetOut, AssetIn, EnvVar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
import mlflow
import optuna
from models.objective import DecisionTreeObjective, SVCObjective

from models.utils import score_models

MLFLOW_HOST = EnvVar("MLFLOW_HOST").get_value()
MLFLOW_PORT = EnvVar("MLFLOW_PORT").get_value()
if MLFLOW_HOST is None:
    raise Exception("MLFLOW_HOST must be set")
if MLFLOW_PORT is None:
    raise Exception("MLFLOW_PORT must be set")


def setup_mlflow():
    mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

    mlflow.set_experiment(experiment_name="Training")


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

    categorical_features = data_type["categorical_features"]
    numeric_features = data_type["numeric_features"]

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
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ """

    lbl = LabelBinarizer()
    y_vec = lbl.fit_transform(y["class"]).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vec, test_size=0.1, random_state=42
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
def svm_optuna(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    data_type: dict[str, list[str]],
):
    setup_mlflow()
    y_test_vec = y_test[y_test.columns[0]].to_numpy()
    y_train_vec = y_train[y_train.columns[0]].to_numpy()
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name="svm_optuna", nested=True):
        # Initialize the Optuna study
        storage = optuna.storages.RDBStorage(
            url="postgresql://optuna_user:optuna_pass@postgres/optuna"
        )
        study = optuna.create_study(
            storage=storage, direction="maximize"
        )

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        objective = SVCObjective(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train_vec,
            y_test=y_test_vec,
            data_type=data_type,
        )
        study.optimize(objective, n_trials=1, show_progress_bar=True)

        mlflow.log_metric("best_balanced_score", study.best_value)
        mlflow.sklearn.autolog(disable=False, log_datasets=False)

        model = objective.create_pipeline()
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)
        
        mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score = (
            score_models(model, X_test, y_test_vec)
        )
        mlflow.log_metric(key="test_f1_score", value=mod_f1_score)
        mlflow.log_metric(key="test_precision_score", value=mod_precision_score)
        mlflow.log_metric(key="test_recall_score", value=mod_recall_score)
        mlflow.log_metric(key="test_balanced_accuracy_score", value=mod_accuracy_score)


@asset(
    ins={
        "X_train": AssetIn(key="X_train"),
        "X_test": AssetIn(key="X_test"),
        "y_train": AssetIn(key="y_train"),
        "y_test": AssetIn(key="y_test"),
        "data_type": AssetIn(key="data_type"),
    }
)
def decision_tree_optuna(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    data_type: dict[str, list[str]],
):
    setup_mlflow()
    mlflow.sklearn.autolog(disable=True)
    y_test_vec = y_test[y_test.columns[0]].to_numpy()
    y_train_vec = y_train[y_train.columns[0]].to_numpy()
    with mlflow.start_run(run_name="decision_tree_optuna", nested=True):
        # Initialize the Optuna study
        storage = optuna.storages.RDBStorage(
            url="postgresql://optuna_user:optuna_pass@postgres/optuna"
        )
        study = optuna.create_study(
            storage=storage, direction="maximize"
        )

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        objective = DecisionTreeObjective(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train_vec,
            y_test=y_test_vec,
            data_type=data_type,
        )
        study.optimize(objective, timeout=60 * 2, show_progress_bar=True)

        mlflow.log_metric("best_balanced_score", study.best_value)
        mlflow.sklearn.autolog(disable=False, log_datasets=False)
        
        model = objective.create_pipeline()
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)
        # mlflow.sklearn.log_model(model,"model")
        mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score = (
            score_models(model, X_test, y_test_vec)
        )
        mlflow.log_metric(key="test_f1_score", value=mod_f1_score)
        mlflow.log_metric(key="test_precision_score", value=mod_precision_score)
        mlflow.log_metric(key="test_recall_score", value=mod_recall_score)
        mlflow.log_metric(key="test_balanced_accuracy_score", value=mod_accuracy_score)

@asset(deps=[svm_optuna,decision_tree_optuna])
def register_best_model():
    setup_mlflow()
    runs = mlflow.search_runs(experiment_names=["Training"],filter_string="metrics.test_f1_score > 0")
    runs = runs.sort_values(["metrics.test_balanced_accuracy_score"],ascending=False)
    best_artifact_uri = runs.iloc[0,runs.columns.get_loc("artifact_uri")]
    best_run_id = runs.iloc[0,runs.columns.get_loc("run_id")]
    model_uri = f"runs:/{best_run_id}{best_artifact_uri.split(best_run_id)[-1]}"
    mlflow.register_model(model_uri=model_uri,name="best_model")

    


train_model_pipeline = define_asset_job(
    name="train_model_pipeline",
    selection=[
        load_data,
        preprocess,
        train_test_splitter,
        svm_optuna,
        decision_tree_optuna,
        register_best_model
    ],
)
