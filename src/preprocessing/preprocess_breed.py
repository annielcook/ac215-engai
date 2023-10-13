import re
import tensorflow as tf
import numpy as np

from google.cloud import storage
from torchvision import transforms
from PIL import Image


def resize_img(blb, proc_bkt, curr_ext):
    local_image_file = 'curr_image' + curr_ext
    blb.download_to_filename(local_image_file)
    image = Image.open(local_image_file)
    convert_tensor = transforms.ToTensor()
    image_tensor = convert_tensor(image)
    image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensors = tf.image.resize(image_tensor, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
    img = tf.image.convert_image_dtype(image_tensors[0], dtype=tf.uint8)
    Image.fromarray(np.array(img)).save(local_image_file)
    destination_blob = proc_bkt.blob(blb.name)
    destination_blob.upload_from_filename(local_image_file)

RAW_BREED_NAME="team-engai-dogs"
RAW_BREED_PREF="dog_breed_dataset/images/Images"
client = storage.Client.from_service_account_json('../secrets/data-service-account.json')
blobs_breed = client.list_blobs(RAW_BREED_NAME, prefix=RAW_BREED_PREF)

PROC_BREED_NAME="team-engai-dogs-processed"
proc_breed = client.get_bucket(PROC_BREED_NAME)


blobs_breed = list(blobs_breed)
print(f'Found {len(blobs_breed)} blobs to resize!')

for blob in blobs_breed:
    if ".DS_Store" not in blob.name:
        if(not blob.name.endswith("/")):
          curr_ext = '.jpg'
          if blob.name.endswith('png'):
              curr_ext = '.png'
        file_name = blob.name.split('/')[-1]
        resize_img(file_name, blob, proc_breed, curr_ext)

print('Resizing complete!')