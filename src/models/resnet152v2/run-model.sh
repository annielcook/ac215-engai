#!/bin/bash
echo "Running model training script..."
pipenv run wandb login $WANDB_KEY
# pipenv run python3 model_training_age_dataset.py
pipenv run python3 model_training_breed_dataset_distillation.py
pipenv shell
