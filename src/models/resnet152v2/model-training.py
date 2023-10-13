# Common 
import os
import numpy as np
import time

# Data
from keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage

# Model
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

# Weights and Biases
import wandb
from wandb.keras import WandbCallback

# Download training data

TENSORIZED_BUCKET_NAME="team-engai-dogs-tensorized"

client = storage.Client.from_service_account_json('../secrets/data-service-account.json')
blobs = client.list_blobs(TENSORIZED_BUCKET_NAME, prefix="dog_breed_dataset/images/Images")

blobs = list(blobs)
print(f'Found {len(blobs)} blobs to train with!')

for blob in blobs:
  print(blob)
  blob.download_to_filename(blob.name)

DATA_PATH = './dog_breed_dataset/images/Images'
BREED_COUNT = 120

# Specify Model Name
name = "DogNetV1"

# Initialize Generator 

gen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=20, 
)

# Load data
train_ds = gen.flow_from_directory(
    DATA_PATH, 
    target_size=(224,224), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True,
    subset='training' 
)

valid_ds = gen.flow_from_directory(
    DATA_PATH, 
    target_size=(224,224), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True,
    subset='validation'
)

# # Pretrained Model
base_model = ResNet152V2(include_top=False, input_shape=(224,224,3), weights='imagenet')
base_model.trainable = False # Freeze the Weights

# # Model 
DogNetV1 = Sequential([
    base_model,
    GlobalAvgPool2D(),
    Dense(224, activation='leaky_relu'),
    Dense(BREED_COUNT, activation='softmax')
], name=name)

DogNetV1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'recall', 'precision', 'f1_score'])


# Initialize W&B

epochs = 100

wandb.init(
    project = "DogNet",
    config = {
        "learning_rate": 0.02,
        "epochs": epochs,
        "architecture": "ResNet152V2",
        "batch_size": 32,
        "model_name": name
    },
    name = DogNetV1.name
)


# # Callbacks
callbacks = [
    WandbCallback(),
    ModelCheckpoint(filepath=f'{name}.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# # Train
start_time = time.time()
DogNetV1.fit_generator(
    train_ds, 
    epochs=epochs, 
    validation_data=valid_ds, 
    callbacks=callbacks, 
    verbose=1
)

execution_time = (time.time() - start_time)/60.0

wandb.config.update({"execution_time": execution_time})
wandb.run.finish()


