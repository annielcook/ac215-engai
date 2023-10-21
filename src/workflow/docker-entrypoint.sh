#!/bin/bash

echo "Container is running!!!"

# Install Google Cloud SDK
# curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-451.0.1-linux-x86_64.tar.gz
# tar -xf google-cloud-cli-451.0.1-linux-x86.tar.gz
# ./google-cloud-sdk/install.sh


# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
# Set GCP Project Details
gcloud config set project $GCP_PROJECT
# Configure GCR
gcloud auth configure-docker gcr.io -q

#/bin/bash
pipenv shell