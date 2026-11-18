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
        """Create the boto3 S3 client from the configured endpoint and credentials."""
        self.client: S3Client = (
            boto3.client(  # pyright: ignore[reportUnknownMemberType,reportUninitializedInstanceVariable]
                service_name="s3",
                endpoint_url=f"http://{self.rustfs_host}:{self.rustfs_port}",
                aws_access_key_id=self.rustfs_access_key,
                aws_secret_access_key=self.rustfs_secret_key,
            )
        )

    def _s3_uri(self, bucket: str, key: str) -> str:
        """Build an S3 URI for an object.

        Args:
            bucket: Name of the S3 bucket.
            key: Object key within the bucket.

        Returns:
            The S3 URI for the object.
        """
        return f"s3://{bucket}/{key}"

    def _storage_options(self) -> dict[str, str]:
        """Return credentials and endpoint settings for pandas S3 access.

        Returns:
            Storage options for pandas Parquet operations.
        """
        return {
            "key": self.rustfs_access_key,
            "secret": self.rustfs_secret_key,
            "endpoint_url": f"http://{self.rustfs_host}:{self.rustfs_port}",
        }

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame as a Parquet object in the configured bucket.

        Args:
            df: DataFrame to save.
            filename: Object key to use for the Parquet file.
        """
        s3_path = self._s3_uri(self.bucket, filename)
        df.to_parquet(s3_path, storage_options=self._storage_options())

    def upload_object(
        self,
        Key: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a local file to the configured S3 bucket.

        Args:
            Key: Destination object key.
            file_path: Path to the local file to upload.
            content_type: MIME type to set on the uploaded object.
        """
        self.client.upload_file(
            Bucket=self.bucket,
            Key=Key,
            Filename=file_path,
            ExtraArgs={"ContentType": content_type},
        )

    def read_object(
        self, Key: str, object_type: Literal["json"]
    ) -> dict:  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        """Read and decode a JSON object from the configured S3 bucket.

        Args:
            Key: Object key to read.
            object_type: Supported object format; currently ``"json"``.

        Returns:
            The decoded JSON object.

        Raises:
            Exception: Propagates an error raised while fetching or decoding
                the object.
        """
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=Key)
            if object_type == "json":
                return json.loads(
                    response["Body"].read().decode()
                )  # pyright: ignore[reportAny]
        except Exception as e:
            print(e)
            raise

    def read_dataframe(self, Key: str) -> pd.DataFrame:
        """Read a Parquet object from the configured S3 bucket.

        Args:
            Key: Object key of the Parquet file.

        Returns:
            The contents of the Parquet file as a DataFrame.
        """
        s3_path = self._s3_uri(self.bucket, Key)

        df = pd.read_parquet(s3_path, storage_options=self._storage_options())

        return df

    def find_object(self, prefix: str) -> ObjectTypeDef:
        """Find the first object whose key starts with the given prefix.

        Args:
            prefix: Prefix used to search the configured bucket.

        Returns:
            Metadata for the first matching object.

        Raises:
            Exception: If the bucket contains no matching objects.
        """
        object_list = self.client.list_objects(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in object_list:
            raise Exception(f"Object {prefix} not found in {self.bucket}")
        if len(object_list) == 0:
            raise Exception(f"Object {prefix} not found in {self.bucket}")
        return object_list["Contents"][0]
