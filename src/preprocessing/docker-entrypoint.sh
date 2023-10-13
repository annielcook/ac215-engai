#!/bin/bash

echo "Preprocessing entrypoint!"

pipenv shell

python3 preprocess_age.py
python3 preprocess_breed.py

