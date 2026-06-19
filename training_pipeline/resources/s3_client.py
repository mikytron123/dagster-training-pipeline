import json
from dataclasses import dataclass
from typing import Literal

import boto3
import pandas as pd
from types_boto3_s3.client import S3Client
from types_boto3_s3.type_defs import ObjectTypeDef


@dataclass()
class S3_Client:
    rustfs_host: str
    rustfs_port: str
    rustfs_access_key: str
    rustfs_secret_key: str
    bucket: str

    def __post_init__(
        self,
    ) -> None:
        self.client: S3Client = boto3.client( #pyright: ignore[reportUnknownMemberType,reportUninitializedInstanceVariable]
            service_name="s3",
            endpoint_url=f"http://{self.rustfs_host}:{self.rustfs_port}",
            aws_access_key_id=self.rustfs_access_key,
            aws_secret_access_key=self.rustfs_secret_key,
        )
    def _s3_uri(self,bucket:str,key:str)->str:
        return f"s3://{bucket}/{key}"

    def _storage_options(self)->dict[str,str]:
        return {
            "key": self.rustfs_access_key,
            "secret": self.rustfs_secret_key,
            "endpoint_url": f"http://{self.rustfs_host}:{self.rustfs_port}"
        }
    def save_dataframe(self, df: pd.DataFrame, filename: str)->None:
        """ saves a dataframe as parquet to s3

        Args:
            df (pd.DataFrame): pandas dataframe to save
            filename (str) : filename to save
        """
        s3_path = self._s3_uri(self.bucket,filename)
        df.to_parquet(s3_path,storage_options=self._storage_options())

    def upload_object(
        self,
        Key: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    )->None:
        self.client.upload_file(
            Bucket=self.bucket,
            Key=Key,
            Filename=file_path,
            ExtraArgs={"ContentType": content_type},
        )

    def read_object(self, Key: str, object_type: Literal["json"])->dict: #pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=Key)
            if object_type == "json":
                return json.loads(response["Body"].read().decode()) #pyright: ignore[reportAny]
        except Exception as e:
            print(e)
            raise
    def read_dataframe(self, Key: str) -> pd.DataFrame:

        s3_path = self._s3_uri(self.bucket,Key)
        
        df = pd.read_parquet(s3_path,storage_options=self._storage_options())
        
        return df

    def find_object(self, prefix: str)->ObjectTypeDef:
        object_list = self.client.list_objects(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in object_list:
            raise Exception(f"Object {prefix} not found in {self.bucket}")
        if len(object_list) == 0:
            raise Exception(f"Object {prefix} not found in {self.bucket}")
        return object_list["Contents"][0]
