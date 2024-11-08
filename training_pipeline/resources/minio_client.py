from typing import Literal
from minio import Minio
import pandas as pd
import os
from io import BytesIO
import pathlib
from minio.datatypes import Object
from dataclasses import dataclass

@dataclass()
class MinioClient:
    minio_host: str
    minio_port: str
    access_key: str
    secret_key: str
    bucket:str

    def __post_init__(
        self,
    ) -> None:
        self.client = Minio(
            endpoint=f"{self.minio_host}:{self.minio_port}",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
        )

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        buff = BytesIO()
        df.to_csv(buff)

        self.client.put_object(
            bucket_name=self.bucket,
            object_name=filename,
            data=buff,
            length=len(buff.getvalue()),
        )

    def upload_object(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ):
        self.client.fput_object(
            bucket_name=self.bucket,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type,
        )

    def download_object(self, object_name: str, file_path: str):
        self.client.fget_object(
            bucket_name=self.bucket, object_name=object_name, file_path=file_path
        )

    def read_object(self, object_name: str, object_type: Literal["dataframe", "json"]):
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket, object_name=object_name
            )
            response = self.client.get_object(
                bucket_name=self.bucket, object_name=object_name, length=stat.size
            )
            if object_type == "dataframe":
                return response.data
            elif object_type == "json":
                return response.json()
        finally:
            response.close()
            response.release_conn()

    def read_dataframe(self, object_name: str) -> pd.DataFrame:
        data = self.read_object(object_name=object_name, object_type="dataframe")
        df = pd.read_parquet(BytesIO(data))
        return df

    def save_and_upload_dataframe(self, df: pd.DataFrame, filename: str):
        pth = pathlib.Path().resolve() / filename
        df.to_parquet(str(pth), index=False)
        self.upload_object(filename, str(pth))

    def find_object(self, prefix: str) -> Object:
        object_list = self.client.list_objects(bucket_name=self.bucket, prefix=prefix)
        for obj in object_list:
            return obj
        raise Exception(f"Object {prefix} not found in {self.bucket}")
