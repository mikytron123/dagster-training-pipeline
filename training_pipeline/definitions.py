from assets import (
    decision_tree_optuna,
    load_data,
    preprocess,
    register_best_model,
    svm_optuna,
    train_model_pipeline,
    train_test_splitter,
)
from resources.s3_parquet_io_manager import S3ParquetIOManager
from resources.s3_resource import S3Resource

from dagster import Definitions, EnvVar

s3_resource = S3Resource(
    rustfs_host=EnvVar("RUSTFS_HOST"),
    rustfs_port=EnvVar("RUSTFS_PORT"),
    rustfs_access_key=EnvVar("RUSTFS_ACCESS_KEY"),
    rustfs_secret_key=EnvVar("RUSTFS_SECRET_KEY"),
    bucket=EnvVar("RUSTFS_BUCKET"),
)

io_manager = S3ParquetIOManager(s3_resource=s3_resource)

defs = Definitions(
    assets=[
        load_data,
        preprocess,
        train_test_splitter,
        svm_optuna,
        decision_tree_optuna,
        register_best_model,
    ],
    jobs=[train_model_pipeline],
    resources={"io_manager": io_manager},
)
