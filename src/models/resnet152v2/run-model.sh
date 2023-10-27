#!/bin/bash

pipenv run wandb login 9dc62923f7261906295e4875a4e598a9b0a91d46
# pipenv run python3 model_training_age_dataset.py
pipenv run python3 model_training_breed_dataset_distillation.py
pipenv shell
