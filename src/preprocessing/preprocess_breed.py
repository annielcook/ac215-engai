import re
import tensorflow as tf
import numpy as np

from google.cloud import storage
from torchvision import transforms
from PIL import Image


def resize_img(fn, blb, proc_bkt):
	blb.download_to_filename(fn)
	image = Image.open(fn)
	convert_tensor = transforms.ToTensor()
	image_tensor = convert_tensor(image)
	image_tensor = image_tensor.permute(1, 2, 0)
	image_tensor = image_tensor.unsqueeze(0)
	image_tensors = tf.image.resize(image_tensor, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
	img = tf.image.convert_image_dtype(image_tensors[0], dtype=tf.uint8)
	Image.fromarray(np.array(img)).save(fn)
	destination_blob = proc_bkt.blob(blb.name)
	destination_blob.upload_from_filename(fn)

RAW_BREED_NAME="team-engai-dogs"
RAW_BREED_PREF="dog_breed_dataset/images/Images"
client = storage.Client.from_service_account_json('../secrets/data-service-accounts.json')
blobs_breed = client.list_blobs(RAW_BREED_NAME, prefix=RAW_BREED_PREF)

PROC_BREED_NAME="team-engai-dogs-processed"
proc_breed = client.get_bucket(PROC_BREED_NAME)


blobs_breed = list(blobs_breed)

for blob in blobs_breed:
  if ".DS_Store" not in blob.name:
      if(not blob.name.endswith("/")):
        file_name = blob.name.split('/')[-1]
        resize_img(file_name, blob, proc_breed)