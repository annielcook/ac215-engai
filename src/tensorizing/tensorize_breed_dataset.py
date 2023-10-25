## This script takes the resized images from the GCP bucket
## and then tensorizes the data
from google.cloud import storage
from PIL import Image
import re
import tensorflow as tf
import os


PROCESSED_BUCKET_NAME="team-engai-dogs-processed"
PROCESSED_BUCKET_NAME_PREFIX="dog_breed_dataset/images/Images"
TENSORIZED_BUCKET_NAME="team-engai-dogs-tensorized"


## function to take in image bytes and a label for the image and tensorizing
## the input
def create_example(image_bytes, label):
    # Create a dictionary with the image data and label
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[224])),
        'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[224])),
        'channel':tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    # Create an example proto
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

## Read resized data from bucket
client = storage.Client.from_service_account_json('secrets/data-service-account.json')
blobs = client.list_blobs(PROCESSED_BUCKET_NAME, prefix=PROCESSED_BUCKET_NAME_PREFIX)
proc_bucket = client.get_bucket(PROCESSED_BUCKET_NAME)

tensor_bucket = client.get_bucket(TENSORIZED_BUCKET_NAME)
blobs = list(blobs)

print(f'Found {len(blobs)} blobs to tensorize!')
breed_to_int = {}
count = 0
for blob in blobs:
  path = os.path.dirname(blob.name)
  breed_name = os.path.basename(path)
  class_label = breed_name[breed_name.index('-') + 1:]
  suffix = ".jpg"
  if re.search(r"png",blob.name):
    suffix = '.png'
  if class_label in breed_to_int:
    class_int_label = breed_to_int[class_label]
  else:
    breed_to_int[class_label] = count
    class_int_label = count
    count += 1
  file_name = blob.name.split('/')[-1].split('.')[0] + "_processed_" + str(class_label) + suffix
  local_file_name = 'curr_image' + suffix
  blob.download_to_filename(local_file_name)
  image =Image.open(local_file_name)
  image_tensor = tf.convert_to_tensor(image)
  img = tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)
  example = create_example(bytes(img), class_int_label)
  destination_blob = tensor_bucket.blob(blob.name)
  destination_blob.upload_from_string(example)

print('Tensorizing complete!')