# Common
import os
import time

# Google
from google.cloud import storage

# Model
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

# Weights and Biases
import wandb
from wandb.keras import WandbCallback


def parse_tfrecord_example(example_proto):
  parsed_example = tf.io.parse_single_example(example_proto, feature_description)

  # Image
  #image = tf.image.decode_jpeg(parsed_example['image'])
  image = tf.io.decode_raw(parsed_example['image'], tf.uint8)
  image.set_shape([num_channels * image_height * image_width])
  image = tf.reshape(image, [image_height, image_width, num_channels])

  # Label
  label = tf.cast(parsed_example['label'], tf.int64)
  label = tf.one_hot(label, num_classes)

  return image, label

# Normalize pixels
def normalize(image, label):
  image = image/255
  return image, label

# Connect to GCS Bucket
TENSORIZED_DATA_BUCKET_NAME="team-engai-dogs-tensorized"
client = storage.Client.from_service_account_json('./secrets/data-service-account.json')
blobs = client.list_blobs(TENSORIZED_DATA_BUCKET_NAME, prefix='dog_breed_dataset/images/Images')


breed_directory_name = 'breed-data'

if not os.path.exists(breed_directory_name):
    os.mkdir(breed_directory_name)


train_directory_name = 'breed-data/train'
validation_directory_name = 'breed-data/validate'

if not os.path.exists(train_directory_name):
    os.mkdir(train_directory_name)

if not os.path.exists(validation_directory_name):
    os.mkdir(validation_directory_name)

breed_to_image_files = {}

for blob in blobs:
  image_file_name = blob.name.split('/')[-1]
  breed_name = blob.name.split('/')[-2]
  if breed_name not in breed_to_image_files:
    breed_to_image_files[breed_name] = []
  breed_to_image_files[breed_name].append(image_file_name)

for breed_name in breed_to_image_files:
  num_images = len(breed_to_image_files[breed_name])
  num_train_images = int(num_images * 0.8)
  for i in range(num_images):
    blob = client.get_bucket(TENSORIZED_DATA_BUCKET_NAME).blob(f'dog_breed_dataset/images/Images/{breed_name}/{breed_to_image_files[breed_name][i]}')
    if i < num_train_images:
      blob.download_to_filename(f'{train_directory_name}/{breed_to_image_files[breed_name][i]}')
    else:
      blob.download_to_filename(f'{validation_directory_name}/{breed_to_image_files[breed_name][i]}')


# Create a dictionary with the image data and label
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'height':tf.io.FixedLenFeature([], tf.int64),
    'width':tf.io.FixedLenFeature([], tf.int64),
    'channel':tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
num_channels = 3
image_height = 224
image_width = 224
num_classes = len(breed_to_image_files)
batch_size = 32

os.chdir('breed-data')


# Read the tfrecord files
train_tfrecord_files = tf.data.Dataset.list_files('train/*')
train_data = train_tfrecord_files.flat_map(tf.data.TFRecordDataset)
train_data = train_data.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# Read the tfrecord files
validate_tfrecord_files = tf.data.Dataset.list_files('validate/*')
validation_data = validate_tfrecord_files.flat_map(tf.data.TFRecordDataset)
validation_data = validation_data.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = validation_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = validation_data.batch(batch_size)
validation_data = validation_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# Pretrained Model
base_model = ResNet152V2(include_top=False, input_shape=(224,224,3), weights='imagenet')
base_model.trainable = False # Freeze the Weights
BREED_COUNT = len(breed_to_image_files)

# Model Name
name = "DogNetV1-breed"

# Model
DogNetV1_breed = Sequential([
    base_model,
    GlobalAvgPool2D(),
    Dense(224, activation='leaky_relu'),
    Dense(BREED_COUNT, activation='softmax')
], name=name)

DogNetV1_breed.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath='DogNetV1_breed.h5', monitor='val_loss', save_best_only=True, verbose=1),
    WandbCallback()
]

epochs = 100

wandb.init(
    project = "DogNet-breed",
    config = {
        "learning_rate": 0.02,
        "epochs": epochs,
        "architecture": "ResNet152V2",
        "batch_size": 32,
        "model_name": name
    },
    name = DogNetV1_breed.name
)


# Train

start_time = time.time()
DogNetV1_breed.fit(
    train_data,
    epochs=epochs,
    validation_data=validation_data,
    callbacks=callbacks,
    verbose=1
)
execution_time = (time.time() - start_time)/60.0

wandb.config.update({"execution_time": execution_time})
wandb.run.finish()




