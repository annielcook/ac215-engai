#!/bin/bash

echo "Preprocessing entrypoint!"

pipenv run python3 preprocess_age.py
pipenv run python3 preprocess_breed.py

