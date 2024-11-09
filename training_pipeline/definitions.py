from dagster import Definitions,EnvVar

from assets import (
    train_model_pipeline,
    load_data,
    preprocess,
    train_test_splitter,
    decision_tree,
    svm,
    svm_optuna
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

io_manager = MinioParquetIOManager(minio_resource=minio_resource, minio_bucket=EnvVar("MINIO_BUCKET"))

defs = Definitions(
    assets=[
        load_data,
        preprocess,
        train_test_splitter,
        decision_tree,
        svm,
        svm_optuna
    ],
    jobs=[train_model_pipeline],
    resources={"io_manager": io_manager},
)
