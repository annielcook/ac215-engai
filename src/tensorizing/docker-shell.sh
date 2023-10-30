#!/bin/bash
usage() { echo "Usage: $0 [-n <string>]" 1>&2; exit 1; }

while getopts ":n:" o; do
    case "${o}" in
        n)
            n=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] ; then
    usage
fi

echo "n = ${n}"

set -e

export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_NAME="team-engai"
export GCP_PROJECT="AC215Project"
export GCP_ZONE="northamerica-northeast2"

# Create the network if we don't have it yet
docker network inspect data-versioning-network >/dev/null 2>&1 || docker network create data-versioning-network

# Build the image based on the Dockerfile
docker build -t tensorizing --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name tensorizing -ti \
--mount type=bind,source="$BASE_DIR",target=/app \
--mount type=bind,source="$SECRETS_DIR",target=/secrets \
-v ~/.gitconfig:/etc/gitconfig \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e LD_PRELOAD=$LD_PRELOAD \
--network data-versioning-network tensorizing

# Below replaced by `mount` above.
# -v "$BASE_DIR":/app \
# -v "$SECRETS_DIR":/secrets \