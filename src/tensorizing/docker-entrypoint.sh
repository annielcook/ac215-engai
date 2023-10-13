#!/bin/bash

echo "Preprocessing entrypoint!"

pipenv shell

python3 tensorize_age_dataset.py
python3 tensorize_breed_dataset.py

