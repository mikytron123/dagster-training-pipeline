from dagster import Definitions, EnvVar

from assets import (
    decision_tree_optuna,
    train_model_pipeline,
    load_data,
    preprocess,
    train_test_splitter,
    svm_optuna,
    register_best_model
)
from resources.minio_resource import MinioResource
from resources.minio_parquet_io_manager import MinioParquetIOManager


minio_resource = MinioResource(
    minio_host=EnvVar("MINIO_HOST"),
    minio_port=EnvVar("MINIO_PORT"),
    access_key=EnvVar("MINIO_ROOT_USER"),
    secret_key=EnvVar("MINIO_ROOT_PASSWORD"),
    bucket=EnvVar("MINIO_BUCKET"),
)

io_manager = MinioParquetIOManager(minio_resource=minio_resource)

defs = Definitions(
    assets=[
        load_data,
        preprocess,
        train_test_splitter,
        svm_optuna,
        decision_tree_optuna,
        register_best_model
    ],
    jobs=[train_model_pipeline],
    resources={"io_manager": io_manager},
)
