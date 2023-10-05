# Common 
import os
import keras
import numpy as np

# Data
from keras.preprocessing.image import ImageDataGenerator

# Model
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

TRAINING_PATH = ''
VALIDATION_PATH = ''
BREED_COUNT = 120

# Initialize Generator 
training_gen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=20, 
)

validation_gen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=20, 
)

# Load data
train_ds = training_gen.flow_from_directory(
    TRAINING_PATH, 
    target_size=(224,224), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True, 
)

valid_ds = validation_gen.flow_from_directory(
    VALIDATION_PATH, 
    target_size=(224,224), 
    class_mode='binary', 
    batch_size=32, 
    shuffle=True, 
)

# # Specify Model Name
name = "DogNetV1"

# # Pretrained Model
base_model = ResNet152V2(include_top=False, input_shape=(224,224,3), weights='imagenet')
base_model.trainable = False # Freeze the Weights

# # Model 
resnet152V2 = Sequential([
    base_model,
    GlobalAvgPool2D(),
    Dense(224, activation='leaky_relu'),
    Dense(BREED_COUNT, activation='softmax')
], name=name)

resnet152V2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'recall', 'precision', 'f1_score'])

# # Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath=f'{name}.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# # Train
resnet152V2.fit_generator(
    train_ds, 
    epochs=100, 
    validation_data=valid_ds, 
    callbacks=callbacks, 
    verbose=1
)