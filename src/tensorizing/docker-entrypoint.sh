#!/bin/bash

echo "Preprocessing entrypoint!"

pipenv run python3 tensorize_age_dataset.py
pipenv run python3 tensorize_breed_dataset.py

