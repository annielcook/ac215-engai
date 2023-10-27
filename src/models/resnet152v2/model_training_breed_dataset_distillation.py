import time
import keras
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

from distiller import Distiller
from model_training_breed_dataset_pruned import (
    split_data,
    EPOCHS,
    MODEL_NAME,
    NUM_CLASSES,
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CHANNELS,
)
from util import evaluate_model


def build_teacher_model(image_height, image_width, num_channels, num_classes):
    base_model = ResNet152V2(
        include_top=False,
        input_shape=(image_height, image_width, num_channels),
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze the Weights

    model = Sequential(
        [
            base_model,
            GlobalAvgPool2D(),
            Dense(224, activation="leaky_relu"),
            Dense(num_classes, activation="softmax"),
        ],
        name="teacher-model",
    )

    return model


def train_teacher_model(teacher_model, train_data, validation_data):
    teacher_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    wandb.init(
        project="DogNet",
        config={
            "learning_rate": 0.02,
            "epochs": EPOCHS,
            "architecture": "ResNet152V2",
            "batch_size": 32,
            "model_name": MODEL_NAME,
        },
        name="teacher-model",
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
    training_results = teacher_model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    execution_time = (time.time() - start_time) / 60.0

    wandb.config.update({"execution_time": execution_time})
    wandb.run.finish()
    evaluate_model(
        teacher_model,
        validation_data,
        training_results.history,
        execution_time,
        0.02,
        BATCH_SIZE,
        EPOCHS,
        "adam",
        save=True,
    )


def build_student_model(
    image_height, image_width, num_channels, num_classes, model_name="student"
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    model = Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=keras.initializers.GlorotUniform(seed=1212),
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=keras.initializers.GlorotUniform(seed=2121),
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
            # keras.layers.Dense(units=32, kernel_initializer=keras.initializers.GlorotUniform(seed=2323)),
            keras.layers.Dense(
                units=num_classes,
                kernel_initializer=keras.initializers.GlorotUniform(seed=3434),
            ),
        ],
        name=model_name,
    )

    return model


def train_student_from_scratch(student_model_scratch, train_data, validation_data):
    student_model_scratch.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train teacher model
    start_time = time.time()
    training_results = student_model_scratch.fit(
        train_data, validation_data=validation_data, epochs=EPOCHS, verbose=1
    )
    execution_time = (time.time() - start_time) / 60.0
    print("Training execution time (mins)", execution_time)

    # Get model training history
    training_history = training_results.history

    # Evaluate model
    evaluate_model(
        student_model_scratch,
        validation_data,
        training_history,
        execution_time,
        0.2,
        BATCH_SIZE,
        EPOCHS,
        "adam",
        save=True,
    )

def distill_teacher_to_student(teacher_model, student_model, train_data, validation_data):
    distiller_model = Distiller(teacher=teacher_model, student=student_model)
    distiller_model.compile(
        optimizer="adam",
        metrics=["accuracy"],
        student_loss_fn=keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        Lambda=0.1,
        temperature=3,
    )

    # Train teacher model
    start_time = time.time()
    training_results = distiller_model.fit(
        train_data, validation_data=validation_data, epochs=EPOCHS, verbose=1
    )
    execution_time = (time.time() - start_time) / 60.0
    print("Training execution time (mins)", execution_time)

    # Get model training history
    training_history = training_results.history

    # Evaluate model
    evaluate_model(
        distiller_model,
        validation_data,
        training_history,
        execution_time,
        0.2,
        BATCH_SIZE,
        EPOCHS,
        "adam",
        save=False,
        loss_metrics=["student_loss","distillation_loss","val_student_loss"],
        acc_metrics=["accuracy","val_accuracy"]
    )



if __name__ == "__main__":
    train_data, validation_data = split_data()
    print(train_data)
    print(validation_data)
    teacher_model = build_teacher_model(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
    train_teacher_model(teacher_model, train_data, validation_data)
    student_model_scratch = build_student_model(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
    print(student_model_scratch.summary())
    distill_teacher_to_student(teacher_model, student_model_scratch, train_data, validation_data)   
    print("Done!")
