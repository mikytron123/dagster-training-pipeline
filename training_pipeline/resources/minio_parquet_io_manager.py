from typing import Any, Union
from dagster import (
    ConfigurableIOManager,
    IOManager,
    InputContext,
    OutputContext,
    ResourceDependency,
)
import pathlib
import json
import pandas as pd
from dagster._utils.cached_method import cached_method

from .minio_resource import MinioResource
from .minio_client import MinioClient


class InnerIOManager(IOManager):
    def __init__(self, minio_client: MinioClient, minio_bucket: str):
        self.minio_client = minio_client
        self.minio_bucket = minio_bucket

    def _get_asset_path(self, context: InputContext | OutputContext) -> str:
        return context.asset_key.path[-1]

    def load_input(self, context: InputContext) -> Any:
        minio_object = self.minio_client.find_object(str(self._get_asset_path(context)))
        object_name = minio_object.object_name
        if object_name.endswith("json"):
            return self.minio_client.read_object(
                object_name=object_name, object_type="json"
            )
        elif object_name.endswith("parquet"):
            return self.minio_client.read_dataframe(object_name=object_name)

    def handle_output(self, context: OutputContext, obj: Union[dict, pd.DataFrame]):
        if isinstance(obj, dict):
            # save as json
            obj_name = f"{self._get_asset_path(context)}.json"
            path = str(pathlib.Path().resolve() / obj_name)
            with open(path, "w") as file:
                json.dump(obj, file)

            self.minio_client.upload_object(
                object_name=obj_name, file_path=path, content_type="text/json"
            )

        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            obj_name = f"{self._get_asset_path(context)}.parquet"
            path = str(pathlib.Path().resolve() / obj_name)
            obj.to_parquet(path, index=False)

            self.minio_client.upload_object(object_name=obj_name, file_path=path)
        elif obj is None:
            print("obj is None, No operation")
        else:
            raise Exception(f"object of type {type(obj)} not supported")
            print(f"object of type {type(obj)} not supported")


class MinioParquetIOManager(ConfigurableIOManager):
    minio_resource: ResourceDependency[MinioResource]
    minio_bucket: str

    @cached_method
    def inner_io_manager(self) -> InnerIOManager:
        return InnerIOManager(
            minio_client=self.minio_resource.get_client(),
            minio_bucket=self.minio_bucket,
        )

    def load_input(self, context: InputContext) -> Any:
        return self.inner_io_manager().load_input(context)

    def handle_output(self, context: OutputContext, obj: Union[dict, pd.DataFrame]):
        self.inner_io_manager().handle_output(context, obj)
