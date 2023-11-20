#!/bin/bash
echo "Container is running!!!"

export GCP_PROJECT="ac215project-399920"
export GCS_MODELS_BUCKET_NAME="team-engai-model-breed"
# tf looks to this variable when its searching for creds to connect to GCP
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/data-service-account.json"

pipenv run wandb login 9dc62923f7261906295e4875a4e598a9b0a91d46

# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file=/secrets/data-service-account.json
# Set GCP Project Details
gcloud config set project $GCP_PROJECT


#/bin/bash
pipenv shell
