/usr/bin/docker-entrypoint.sh server /data --console-address ":9001" &

pid=$!

sleep 10

echo 'setting up minio'
mc alias set myminio http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
mc mb myminio/${MINIO_BUCKET}
mc mb myminio/mlflow
echo 'done setting up minio'

wait $pid
