export IMAGE_NAME=model-deployment-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="ac215project-399920"
export GCS_MODELS_BUCKET_NAME="team-engai-model-breed"

# Source .env to get the API key
source ../../.env

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
-e WANDB_KEY=$WANDB_KEY \
--network data-versioning-network $IMAGE_NAME

