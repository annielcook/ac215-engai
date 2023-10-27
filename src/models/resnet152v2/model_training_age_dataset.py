# Common
import os
import time

# Google
from google.cloud import storage

# Model
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

# Weights and Biases
import wandb
from wandb.keras import WandbCallback

from util import get_data

DATA_DIRECTORY_NAME = "age-data"
TRAIN_DIRECTORY_NAME = "age-data/train"
VALIDATION_DIRECTORY_NAME = "age-data/validate"
NUM_CLASSES = 3
MODEL_NAME = "DogNetV1-age"


def make_required_directories():
    if not os.path.exists(DATA_DIRECTORY_NAME):
        os.mkdir(DATA_DIRECTORY_NAME)
    if not os.path.exists(TRAIN_DIRECTORY_NAME):
        os.mkdir(TRAIN_DIRECTORY_NAME)
    if not os.path.exists(VALIDATION_DIRECTORY_NAME):
        os.mkdir(VALIDATION_DIRECTORY_NAME)


def split_data():
    # Connect to GCS Bucket
    TENSORIZED_DATA_BUCKET_NAME = f"team-engai-dogs-tensorized{os.getenv('PERSON')}"
    client = storage.Client.from_service_account_json(
        "../../secrets/data-service-account.json"
    )

    classes = ["Adult", "Senior", "Young"]
    for class_ in classes:
        count = 0
        blobs = client.list_blobs(
            TENSORIZED_DATA_BUCKET_NAME,
            prefix="dog_age_dataset/Expert_Train/Expert_TrainEval/" + class_,
        )
        total_blobs = sum(1 for _ in blobs)
        split_point = int(total_blobs * 0.8)
        blobs = client.list_blobs(
            TENSORIZED_DATA_BUCKET_NAME,
            prefix="dog_age_dataset/Expert_Train/Expert_TrainEval/" + class_,
        )
        for blob in blobs:
            if count < split_point:
                blob.download_to_filename(
                    f'{TRAIN_DIRECTORY_NAME}/{blob.name.split("/")[-1]}'
                )
            else:
                blob.download_to_filename(
                    f'{VALIDATION_DIRECTORY_NAME}/{blob.name.split("/")[-1]}'
                )
            count += 1

    num_channels = 3
    image_height = 224
    image_width = 224
    batch_size = 32

    os.chdir("age-data")

    # Read the tfrecord files
    train_tfrecord_files = tf.data.Dataset.list_files("train/*")
    train_data = get_data(
        train_tfrecord_files,
        batch_size=batch_size,
        num_channels=num_channels,
        num_classes=NUM_CLASSES,
        image_height=image_height,
        image_width=image_width,
    )
    # Read the tfrecord files
    validate_tfrecord_files = tf.data.Dataset.list_files("validate/*")
    validation_data = get_data(
        validate_tfrecord_files,
        batch_size=batch_size,
        num_channels=num_channels,
        num_classes=NUM_CLASSES,
        image_height=image_height,
        image_width=image_width,
    )

    return train_data, validation_data


def build_and_train_model(train_data, validation_data):
    # Pretrained Model
    base_model = ResNet152V2(
        include_top=False, input_shape=(224, 224, 3), weights="imagenet"
    )
    base_model.trainable = False  # Freeze the Weights

    # Model
    DogNetV1_age = Sequential(
        [
            base_model,
            GlobalAvgPool2D(),
            Dense(224, activation="leaky_relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name=MODEL_NAME,
    )

    DogNetV1_age.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    epochs = 100
    wandb.init(
        project="DogNet-breed",
        config={
            "learning_rate": 0.02,
            "epochs": epochs,
            "architecture": "ResNet152V2",
            "batch_size": 32,
            "model_name": MODEL_NAME,
        },
        name=DogNetV1_age.name,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="DogNetV1_age.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        WandbCallback(),
    ]

    # Train
    start_time = time.time()
    DogNetV1_age.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    execution_time = (time.time() - start_time) / 60.0

    wandb.config.update({"execution_time": execution_time})
    wandb.run.finish()


if __name__ == "__main__":
    make_required_directories()
    train_data, validation_data = split_data()
    build_and_train_model(train_data, validation_data)
