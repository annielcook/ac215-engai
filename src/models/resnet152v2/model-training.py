# Common 
import os
import time

# Data
from google.cloud import storage

# Model
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D

# Callbacks
from keras.callbacks import ModelCheckpoint

# Transfer Learning Models
from tensorflow.keras.applications import ResNet152V2

# Weights and Biases
import wandb
from wandb.keras import WandbCallback


# Connect to GCS Bucket
TENSORIZED_DATA_BUCKET_NAME="team-engai-dogs-tensorized"
client = storage.Client.from_service_account_json('../secrets/data-service-account.json')
blobs = client.list_blobs(TENSORIZED_DATA_BUCKET_NAME, prefix='dog_breed_dataset/images/Images')

print("Blobs:")
n_files = 0
for blob in blobs:
  if not os.path.exists("train"):
    os.makedirs("train")

  filename = blob.name.split('/')[-1]
  blob.download_to_filename('train/' + filename)
  n_files += 1

print("Total files: " + str(n_files))

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
num_classes = 80
batch_size = 32

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
  print(label.shape)
  label = tf.reshape(label, [-1, num_classes])

  return image, label

# Normalize pixels
def normalize(image, label):
  image = image/255
  return image, label

# Read the tfrecord files
tfrecord_files = tf.data.Dataset.list_files('train/*')

tfrecord_files = tfrecord_files.shuffle(buffer_size=n_files)

validation_ratio = 0.2

num_validation_files = int(validation_ratio * n_files)

train_tfrecord_files = tfrecord_files.skip(num_validation_files)
validation_tfrecord_files = tfrecord_files.take(num_validation_files)


train_data = train_tfrecord_files.flat_map(tf.data.TFRecordDataset)
train_data = train_data.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
print(train_data)
train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)

validation_data = validation_tfrecord_files.flat_map(tf.data.TFRecordDataset)
validation_data = validation_data.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = validation_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = validation_data.batch(batch_size)
validation_data = validation_data.prefetch(buffer_size=tf.data.AUTOTUNE)

BREED_COUNT = 80

# Specify Model Name
name = "DogNetV1"


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

DogNetV1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


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
    #WandbCallback(),
    ModelCheckpoint(filepath=f'{name}.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# # Train
start_time = time.time()
DogNetV1.fit(
    train_data, 
    epochs=epochs, 
    validation_data=validation_data, 
    callbacks=callbacks, 
    verbose=1
)

execution_time = (time.time() - start_time)/60.0

wandb.config.update({"execution_time": execution_time})
wandb.run.finish()


