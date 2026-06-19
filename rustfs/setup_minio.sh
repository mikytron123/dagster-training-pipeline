#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# MinIO / AWS CLI Bootstrap Script
# ─────────────────────────────────────────────────────────────────
# Required env vars:
#   MINIO_ENDPOINT   e.g. http://minio:9000
#   AWS_ACCESS_KEY   MinIO root / service account access key
#   AWS_SECRET_KEY   MinIO root / service account secret key
#
# Optional env vars (have defaults):
#   BUCKET_1         first bucket  (default: bucket-one)
#   BUCKET_2         second bucket (default: bucket-two)
# ─────────────────────────────────────────────────────────────────


: "${RUSTFS_ACCESS_KEY:?  ❌  AWS_ACCESS_KEY is required}"
: "${RUSTFS_SECRET_KEY:?  ❌  AWS_SECRET_KEY is required}"

BUCKET_1="${BUCKET_1}"
BUCKET_2="${BUCKET_2}"

echo "────────────────────────────────────────────"
echo " Configuring AWS CLI → MinIO"
echo " Endpoint : $AWS_ENDPOINT_URL"
echo " Bucket 1 : $BUCKET_1"
echo " Bucket 2 : $BUCKET_2"
echo "────────────────────────────────────────────"

# ── 1. Write credentials ──────────────────────────────────────────
aws configure set aws_access_key_id     "$RUSTFS_ACCESS_KEY"
aws configure set aws_secret_access_key "$RUSTFS_SECRET_KEY"
aws configure set default.region        "us-east-1"   # MinIO ignores region but CLI requires one
aws configure set default.output        "json"

echo "[OK] Credentials configured."

# ── 2. Thin wrapper so every call goes to MinIO ───────────────────
# s3() {
#   aws --endpoint-url "$MINIO_ENDPOINT" "$@"
# }

# ── 3. Idempotent bucket creation ─────────────────────────────────
create_bucket() {
  local name="$1"
  if aws s3 ls "s3://$name" 2>/dev/null; then
    echo "[SKIP] Bucket already exists: $name"
  else
    aws s3 mb "s3://$name"
    echo "[OK]   Bucket created: $name"
  fi
}
echo "Creating buckets"
create_bucket "$BUCKET_1"
create_bucket "$BUCKET_2"

# ── 4. Confirm ────────────────────────────────────────────────────
echo ""
echo "Buckets in MinIO:"
aws s3api list-buckets --query "Buckets[].Name" --output table

echo ""
echo "Done!"