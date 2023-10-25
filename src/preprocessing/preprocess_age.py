import re

from google.cloud import storage

from util import resize_img


RAW_AGE_NAME="team-engai-dogs"
RAW_AGE_PREF="dog_age_dataset/Expert_Train/Expert_TrainEval"
client = storage.Client.from_service_account_json('secrets/data-service-account.json')
blobs_age = client.list_blobs(RAW_AGE_NAME, prefix=RAW_AGE_PREF)

PROC_AGE_NAME="team-engai-dogs-processed"
proc_age = client.get_bucket(PROC_AGE_NAME)

blobs_age = list(blobs_age)
print(f'Found {len(blobs_age)} blobs to resize!')

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
        curr_ext = '.jpg'
        if blob.name.endswith('png'):
            curr_ext = '.png'
        file_name = blob.name.split('/')[-1].split('.')[0] + "_" + label + curr_ext
        resize_img(blob, proc_age, curr_ext)
  except Exception as e:
    print("got exception: " + str(e))
    continue

print('Resizing complete!')