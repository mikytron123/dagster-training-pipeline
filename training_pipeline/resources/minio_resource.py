from dagster import ConfigurableResource

from .minio_client import MinioClient


class MinioResource(ConfigurableResource):
    minio_host: str
    minio_port: str
    access_key: str
    secret_key: str
    bucket: str

    def get_client(self) -> MinioClient:
        return MinioClient(
            minio_host=self.minio_host,
            minio_port=self.minio_port,
            access_key=self.access_key,
            secret_key=self.secret_key,
            bucket=self.bucket,
        )
