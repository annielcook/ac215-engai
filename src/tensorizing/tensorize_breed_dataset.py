from google.cloud import storage
import re
import tensorflow as tf
import os

PROCESSED_BUCKET_NAME=f"team-engai-dogs-processed{os.getenv('PERSON')}"
PROCESSED_BUCKET_NAME_PREFIX="dog_breed_dataset/images/Images"
TENSORIZED_BUCKET_NAME=f"team-engai-dogs-tensorized{os.getenv('PERSON')}"

def create_tensorized_file(image_bytes, label):
    # Create a dictionary with the image data and label
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])),
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
photo_count = 0
curr_breed = ''

if not os.path.isdir('images'):
    os.mkdir('images')

curr_path = 'images/local_image' + str(photo_count)
for blob in blobs:
  path = os.path.dirname(blob.name)
  breed_name = os.path.basename(path)

  if photo_count == 0:
    curr_breed = breed_name

  if (photo_count % 32 == 0 and photo_count > 0) or breed_name != curr_breed :
    destination_blob = tensor_bucket.blob(path+'/local_image' + str(photo_count))
    with open(curr_path, 'rb') as f:
      destination_blob.upload_from_file(f)
    curr_path = 'images/local_image' + str(photo_count)

  if breed_name != curr_breed:
    curr_breed =  breed_name

  class_label = breed_name[breed_name.index('-') + 1:]
  suffix = ".jpg"
  if re.search(r"png",blob.name):
    suffix = ".png"
  if class_label in breed_to_int:
    class_int_label = breed_to_int[class_label]
  else:
    breed_to_int[class_label] = count
    class_int_label = count
    count += 1
  file_name = blob.name.split('/')[-1].split('.')[0] + "_processed_" + str(class_label) + suffix
  local_file_name = 'curr_image' + suffix
  blob.download_to_filename(local_file_name)
  image = tf.io.read_file(local_file_name)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.uint8)
  example = create_tensorized_file(image, class_int_label)

  with tf.io.TFRecordWriter(curr_path) as writer:
    writer.write(example)
  photo_count += 1

destination_blob = tensor_bucket.blob('local_image' + str(photo_count))
file_contents = tf.io.read_file(curr_path)
destination_blob.upload_from_string(file_contents)

print('Breed dataset tensorizing complete!')