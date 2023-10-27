import os
from google.cloud import storage

from util import resize_img_and_save_locally, upload_image_to_bucket, fetch_or_make_bucket


RAW_BREED_NAME="team-engai-dogs"
RAW_BREED_PREF="dog_breed_dataset/images/Images"
client = storage.Client.from_service_account_json('secrets/data-service-account.json')
blobs_breed = client.list_blobs(RAW_BREED_NAME, prefix=RAW_BREED_PREF)

PROC_BREED_NAME=f"team-engai-dogs-processed{os.getenv('PERSON')}"
proc_breed = fetch_or_make_bucket(PROC_BREED_NAME)


blobs_breed = list(blobs_breed)
print(f'Found {len(blobs_breed)} blobs to resize!')

for blob in blobs_breed:
    if ".DS_Store" not in blob.name:
        if(not blob.name.endswith("/")):
          curr_ext = '.jpg'
          if blob.name.endswith('png'):
              curr_ext = '.png'
        file_name = blob.name.split('/')[-1]
        local_image = resize_img_and_save_locally(blob, curr_ext)
        upload_image_to_bucket(blob, proc_breed, local_image)

print('Resizing complete!')