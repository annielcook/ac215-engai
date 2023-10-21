
echo "Model Training Entry Point!"

pipenv shell

wandb login 9dc62923f7261906295e4875a4e598a9b0a91d46
python model-training.py

