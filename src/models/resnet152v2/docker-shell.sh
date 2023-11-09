#!/bin/bash

set -e

export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export GCS_BUCKET_NAME="team-engai"
export GCP_PROJECT="AC215Project"
export GCP_ZONE="northamerica-northeast2"

# Load environment variables from single .env file in top level directory
source ../../../.env

# Create the network if we don't have it yet
docker network inspect data-versioning-network  >/dev/null 2>&1 || docker network create data-versioning-network 

# Build the image based on the Dockerfile
docker build -t model-training --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name model-training -ti \
--mount type=bind,source="$BASE_DIR",target=/app \
--mount type=bind,source="$SECRETS_DIR",target=/secrets \
-v ~/.gitconfig:/etc/gitconfig \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e WANDB_KEY=$WANDB_KEY \
--network data-versioning-network  model-training

# Below replaced by `mount` above.
# -v "$BASE_DIR":/app \
# -v "$SECRETS_DIR":/secrets \