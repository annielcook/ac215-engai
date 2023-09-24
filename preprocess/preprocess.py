from google.cloud import storage
from io import StringIO
import json

RAW_DATA_BUCKET_NAME="team-engai"
PROCEED_DATA_BUCKET_NAME="team-engai-processed-data"
FIELDS=['class_description', 'class_name', 'fields', 'methods_info']
PYTHON_PROCESSED_DATA_FILE = 'ClassEval_data_processed.json'

## initialize client and get blobs from bucket 
client = storage.Client.from_service_account_json('secrets/data-service-account.json')
bucket = client.get_bucket(RAW_DATA_BUCKET_NAME)
blobs = bucket.list_blobs()

finalArr = []

## for each blob keep the fields mentioned in field
for blob in blobs:
    blob = bucket.blob(blob.name)
    blob = blob.download_as_string()
    blob = blob.decode('utf-8')
    blob = StringIO(blob)  #
    jsonArray = json.loads(blob.getvalue()) 
    
    counter = 0 
    newData = {}
    for data in jsonArray:
        for field in FIELDS:
            newData[field] = data[field]

        finalArr.append(newData)

## write processed data to processed data bucket
processed_data_bucket = client.get_bucket(PROCEED_DATA_BUCKET_NAME)
blob = processed_data_bucket.blob(PYTHON_PROCESSED_DATA_FILE)

blob.upload_from_string(str(finalArr))
