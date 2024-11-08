from dagster import Definitions
import os

from assets import (
    train_model_pipeline,
    load_data,
    preprocess,
    train_test_splitter,
    decision_tree,
    svm
)
from resources.minio_resource import MinioResource
from resources.minio_parquet_io_manager import MinioParquetIOManager


ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_HOST = os.getenv("MINIO_HOST", default="localhost")
MINIO_PORT = os.getenv("MINIO_PORT")
BUCKET = os.getenv("MINIO_BUCKET")


minio_resource = MinioResource(
    minio_host=MINIO_HOST,
    minio_port=MINIO_PORT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    bucket=BUCKET,
)

io_manager = MinioParquetIOManager(minio_resource=minio_resource, minio_bucket=BUCKET)

defs = Definitions(
    assets=[
        load_data,
        preprocess,
        train_test_splitter,
        decision_tree,
        svm
    ],
    jobs=[train_model_pipeline],
    resources={"io_manager": io_manager},
)
