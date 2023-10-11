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

RAW_AGE_NAME="team-engai-dogs"
RAW_AGE_PREF="dog_age_dataset/Expert_Train/Expert_TrainEval"
client = storage.Client.from_service_account_json('../secrets/data-service-accounts.json')
blobs_age = client.list_blobs(RAW_AGE_NAME, prefix=RAW_AGE_PREF)

PROC_AGE_NAME="team-engai-dogs-processed"
proc_age = client.get_bucket(PROC_AGE_NAME)

blobs_age = list(blobs_age)
for blob in blobs_age:
  try:
    if ".DS_Store" not in blob.name:
      if re.search(r"Adult",blob.name):
        print("Adult")
        label = "Adult"
      elif re.search(r"Senior",blob.name):
        print("Senior")
        label = "Senior"
      elif re.search(r"Young",blob.name):
        print("Young")
        label = "Young"
      if(not blob.name.endswith("/")):
        file_name = blob.name.split('/')[-1].split('.')[0] + "_" + label + ".jpg"
        resize_img(file_name, blob, proc_age)
  except:
    print("got exception" + blob.name)
    continue