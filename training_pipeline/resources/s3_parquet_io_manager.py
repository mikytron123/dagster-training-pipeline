import json
import pathlib
from typing import override

import pandas as pd
from dagster._utils.cached_method import cached_method

from dagster import (
    ConfigurableIOManager,
    InputContext,
    IOManager,
    OutputContext,
    ResourceDependency,
)

from .s3_client import S3_Client
from .s3_resource import S3Resource


class InnerIOManager(IOManager):
    def __init__(self, s3_client: S3_Client):
        self.s3_client: S3_Client = s3_client

    def _get_asset_path(self, context: InputContext | OutputContext) -> str:
        return context.asset_key.path[-1]

    @override
    def load_input(
        self, context: InputContext
    ) -> (
        pd.DataFrame | dict | None
    ):  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        s3_object = self.s3_client.find_object(self._get_asset_path(context))

        Key = s3_object["Key"]  # type: ignore

        if Key.endswith("json"):
            return self.s3_client.read_object(
                Key=Key, object_type="json"
            )  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        elif Key.endswith("parquet"):
            return self.s3_client.read_dataframe(Key=Key)

    @override
    def handle_output(
        self,
        context: OutputContext,
        obj: (
            dict | pd.DataFrame | None
        ),  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
    ):
        if isinstance(obj, dict):
            # save as json
            obj_name = f"{self._get_asset_path(context)}.json"
            path = str(pathlib.Path().resolve() / obj_name)
            with open(path, "w") as file:
                json.dump(obj, file)

            self.s3_client.upload_object(
                Key=obj_name, file_path=path, content_type="text/json"
            )

        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            obj_name = f"{self._get_asset_path(context)}.parquet"
            self.s3_client.save_dataframe(obj, obj_name)

        elif obj is None:
            print("obj is None, No operation")
        else:
            raise Exception(f"object of type {type(obj)} not supported")
            print(f"object of type {type(obj)} not supported")


class S3ParquetIOManager(ConfigurableIOManager):
    s3_resource: ResourceDependency[S3Resource]

    @cached_method
    def inner_io_manager(self) -> InnerIOManager:
        return InnerIOManager(s3_client=self.s3_resource.get_client())

    @override
    def load_input(
        self, context: InputContext
    ) -> (
        dict | pd.DataFrame | None
    ):  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        return self.inner_io_manager().load_input(
            context
        )  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

    @override
    def handle_output(
        self,
        context: OutputContext,
        obj: (
            dict | pd.DataFrame | None
        ),  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
    ):
        self.inner_io_manager().handle_output(
            context, obj
        )  # pyright: ignore[reportUnknownMemberType]
