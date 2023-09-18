from google.cloud import storage
from io import StringIO
import json

RAW_DATA_BUCKET_NAME="team-engai"
PROCEED_DATA_BUCKET_NAME="team-engai-processed-data"
PYTHON_PROCESSED_DATA_FILE = 'sql_create_context_v4_processed.json'
FIELDS=['question', 'context']

## initialize client and get blobs from bucket 
client = storage.Client.from_service_account_json('src/preprocessing/secrets/data-service-account.json')
bucket = client.get_bucket(RAW_DATA_BUCKET_NAME)
blobs = bucket.list_blobs()

finalArr = []

## for each blob keep the fields mentioned in field
for blob in blobs:

    if blob.name == 'sql_create_context_v4.json':
        blob = bucket.blob(blob.name)
        blob = blob.download_as_string()
        blob = blob.decode('utf-8')
        blob = StringIO(blob)  #
        jsonArray = json.loads(blob.getvalue()) 
        
        counter = 0 
        
        for data in jsonArray:
            newData = {}
            for field in FIELDS:
                newData[field] = data[field]

            finalArr.append(newData)


## write processed data to processed data bucket
processed_data_bucket = client.get_bucket(PROCEED_DATA_BUCKET_NAME)
blob = processed_data_bucket.blob(PYTHON_PROCESSED_DATA_FILE)

blob.upload_from_string(json.dumps(finalArr))