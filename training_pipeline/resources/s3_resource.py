from dagster import ConfigurableResource

from .s3_client import S3_Client


class S3Resource(ConfigurableResource[S3_Client]):
    rustfs_host: str
    rustfs_port: str
    rustfs_access_key: str
    rustfs_secret_key: str
    bucket: str

    def get_client(self) -> S3_Client:
        """Create an S3 client using this resource's configuration.

        Returns:
            An S3 client configured for the resource's bucket and endpoint.
        """
        return S3_Client(
            rustfs_host=self.rustfs_host,
            rustfs_port=self.rustfs_port,
            rustfs_access_key=self.rustfs_access_key,
            rustfs_secret_key=self.rustfs_secret_key,
            bucket=self.bucket,
        )
