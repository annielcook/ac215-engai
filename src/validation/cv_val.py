from sklearn.model_selection import train_test_split
from google.cloud import storage
from io import StringIO
import json

PROCESSED_DATA_BUCKET_NAME="team-engai-processed-data"
TRAINING_DATA_BUCKET_NAME="team-engai-training-data"
TESTING_DATA_BUCKET_NAME="team-engai-testing-data"
FIELDS=['class_description', 'class_name', 'fields', 'methods_info']
PYTHON_PROCESSED_DATA_FILE = 'ClassEval_data_processed.json'

# initialize client and get blobs from bucket 
client = storage.Client.from_service_account_json('secrets/compute-service-account.json')
bucket = client.get_bucket(PROCESSED_DATA_BUCKET_NAME)
blobs = bucket.list_blobs()

all_data = []
# for each blob keep the fields mentioned in field
for blob in blobs:
    if blob.name == PYTHON_PROCESSED_DATA_FILE:
        blob = bucket.blob(blob.name)
        blob = blob.download_as_string()
        blob = blob.decode('utf-8')
        blob = StringIO(blob)
        json_array = json.loads(blob.getvalue()) 
        
        counter = 0 
        new_data = {}
        for data in json_array:
            for field in FIELDS:
                new_data[field] = data[field]

            all_data.append(new_data)

# Train with 90% of samples, test with 10% .
training_set, testing_set = train_test_split(all_data, test_size=0.1)

training_data_bucket = client.get_bucket(TRAINING_DATA_BUCKET_NAME)
training_blob = training_data_bucket.blob(PYTHON_PROCESSED_DATA_FILE)
training_blob.upload_from_string(str(training_set))

testing_data_bucket = client.get_bucket(TESTING_DATA_BUCKET_NAME)
testing_blob = testing_data_bucket.blob(PYTHON_PROCESSED_DATA_FILE)
testing_blob.upload_from_string(str(testing_set))