# Common
import os
import time
import numpy as np

# Data
from google.cloud import storage

# Model
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

# Model Optimization
import tensorflow_model_optimization as tfmot

# Weights and Biases
import wandb
from wandb.keras import WandbCallback

from util import get_data

EPOCHS = 100
MODEL_NAME = "DogNetV1"
NUM_CLASSES = 80
BATCH_SIZE = 32
NUM_IMAGES = 68449    
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3


def get_blobs():
    # Connect to GCS Bucket
    TENSORIZED_DATA_BUCKET_NAME = f"team-engai-dogs-tensorized{os.getenv('PERSON')}"
    client = storage.Client.from_service_account_json(
        "../../secrets/data-service-account.json"
    )
    return client.list_blobs(
        TENSORIZED_DATA_BUCKET_NAME, prefix="dog_breed_dataset/images/Images"
    )


def split_data():
    blobs = get_blobs()

    n_files = 0
    for blob in blobs:
        if not os.path.exists("train"):
            os.makedirs("train")

        filename = blob.name.split("/")[-1]
        blob.download_to_filename("train/" + filename)
        n_files += 1

    print("Total files: " + str(n_files))

    # Read the tfrecord files
    tfrecord_files = tf.data.Dataset.list_files("train/*")
    tfrecord_files = tfrecord_files.shuffle(buffer_size=n_files)

    validation_ratio = 0.2
    num_validation_files = int(validation_ratio * n_files)

    train_tfrecord_files = tfrecord_files.skip(num_validation_files)
    validation_tfrecord_files = tfrecord_files.take(num_validation_files)

    train_data = get_data(
        train_tfrecord_files,
        batch_size=BATCH_SIZE,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )
    validation_data = get_data(
        validation_tfrecord_files,
        batch_size=BATCH_SIZE,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )
    return train_data, validation_data


def build_and_train_model(train_data, validation_data):
    end_step = np.ceil(NUM_IMAGES / BATCH_SIZE).astype(np.int32) * EPOCHS

    # # Pretrained Model
    base_model = ResNet152V2(
        include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), weights="imagenet"
    )
    base_model.trainable = False  # Freeze the Weights

    # # Model
    DogNetV1 = Sequential(
        [
            base_model,
            GlobalAvgPool2D(),
            Dense(224, activation="leaky_relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name=MODEL_NAME,
    )

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step
        )
    }

    DogNetV1_for_pruning = prune_low_magnitude(DogNetV1, **pruning_params)
    DogNetV1_for_pruning.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Initialize W&B

    wandb.init(
        project="DogNet",
        config={
            "learning_rate": 0.02,
            "epochs": EPOCHS,
            "architecture": "ResNet152V2",
            "batch_size": 32,
            "model_name": MODEL_NAME,
        },
        name=DogNetV1.name,
    )

    # # Callbacks
    callbacks = [
        WandbCallback(),
        ModelCheckpoint(
            filepath=f"{MODEL_NAME}.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tfmot.sparsity.keras.UpdatePruningStep(),
        EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=2,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0,
        ),
    ]

    # # Train
    start_time = time.time()
    DogNetV1_for_pruning.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    execution_time = (time.time() - start_time) / 60.0

    wandb.config.update({"execution_time": execution_time})
    wandb.run.finish()


if __name__ == "__main__":
    train_data, validation_data = split_data()
    build_and_train_model(train_data, validation_data)
