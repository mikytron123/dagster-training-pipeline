FROM amazon/aws-cli:latest

# ── Required env vars (must be supplied at runtime) ──

# ── Optional: override to name your buckets ──
ENV BUCKET_1="mlflow"
ENV BUCKET_2="temp-storage"

WORKDIR /app

COPY setup_minio.sh .
RUN chmod +x setup_minio.sh

ENTRYPOINT ["/app/setup_minio.sh"]