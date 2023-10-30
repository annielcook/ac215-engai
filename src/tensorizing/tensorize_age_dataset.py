from google.cloud import storage
import tensorflow as tf
import os


PROCESSED_BUCKET_NAME=f"team-engai-dogs-processed{os.getenv('PERSON')}"
PROCESSED_BUCKET_NAME_PREFIX="dog_age_dataset/Expert_Train/Expert_TrainEval"
TENSORIZED_BUCKET_NAME=f"team-engai-dogs-tensorized{os.getenv('PERSON')}"

## function to take in image bytes and a label for the image and tensorizing
## the input
def create_tensorized_file(image, label):
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
age_to_int = {}
count = 0
photo_count = 0
curr_age = ''

if not os.path.isdir('age_images'):
    os.mkdir('age_images')

curr_path = 'age_images/local_image' + str(photo_count)
last_class_int_label = 0
class_int_label = 0
writer = tf.io.TFRecordWriter(curr_path)
for blob in blobs:
  path = os.path.dirname(blob.name)
  age_name = os.path.basename(path)

  ## if number of photos in current file is 32 or class (age) has changed
  ## write current file and initialize new batch
  if (photo_count % 32 == 0 or last_class_int_label != class_int_label) and photo_count > 0 :
    destination_blob = tensor_bucket.blob(path+'/tensorized_image_batch_file_' + str(photo_count))
    writer.close()
    with open(curr_path, 'rb') as f:
      print("uploading :" + curr_path)
      destination_blob.upload_from_file(f)
    curr_path = 'age_images/local_image' + str(photo_count)
    writer = tf.io.TFRecordWriter(curr_path)

  ## find class label
  if age_name == 'Adult':
    class_int_label = 0
  elif age_name == 'Senior':
    class_int_label = 1
  else:
    class_int_label = 2

  ## download file locally, tensorize and place in tensorize batch file
  last_class_int_label = class_int_label
  file_name = blob.name.split('/')[-1].split('.')[0] + "_processed_" + str(class_int_label)
  local_file_name = 'curr_image'
  blob.download_to_filename(local_file_name)
  image = tf.io.read_file(local_file_name)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.uint8)
  example = create_tensorized_file(image, class_int_label)

  writer.write(example)
  photo_count += 1

destination_blob = tensor_bucket.blob('local_image' + str(photo_count))
file_contents = tf.io.read_file(curr_path)
destination_blob.upload_from_string(file_contents)

print('Age dataset tensorizing complete!')
