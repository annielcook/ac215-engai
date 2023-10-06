Team EngAi Milestone 2 Deliverable
==============================

AC215 - Milestone2

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      └── src
            ├── preprocessing
            │   ├── Dockerfile
            │   ├── preprocess.py
            │   └── requirements.txt
            └── validation
                  ├── Dockerfile
                  ├── cv_val.py
                  └── requirements.txt


--------
# AC215 - Milestone 3 - mApp

**Team Members**
Nevil George, Juan Pablo Heusser, Curren Iyer, Annie Landefeld, Abhijit Pujare

**Group Name**
EngAi Group

**Project**
In this project, we aim to build an application that can predict a dog's breed and age using a photo.  

### Milestone 3 ###

In this milestone, we extended the pipeline with 3 main changes:
   (1) Update datasets since we changed our idea from Milestone 2
   (2) Converted our data from images (.jpg, .png) to TFRecords
   (3) Trained a first iteration of our model using TensorFlow and the existing ResNet model.

We gathered 2 datasets from the following sources:

Kaggle Stanford Dogs - https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
 - A dataset of dog pictures with their breeds.

Kaggle Dog Age - https://www.kaggle.com/datasets/user164919/the-dogage-dataset
 - A dataset of dog pictures with their ages. 

**Preprocess container**
- This container reads 22.9MB of data, deletes fields that are not relevant to our application, and stores it back to GCP.
- Input to this container is source and destincation GCS location, parameters for deleting fields, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we preprocess the FudanSELab data set. We read in the raw data from the source GCS bucket, and we turn it into a JSON file. We only kept the following data fields: 'class_description', 'class_name', 'fields', 'methods_info' which are relevant to our application. 

(2) `src/preprocessing/preprocesssql.py`  - Here we preprocess the SQL Create Context data set. We read in the raw data from the source GCS bucket, and we turn it into a JSON file. We only kept the following data fields: 'question, 'context' which are relevant for fine-tuning our application model.

(3) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - Google Cloud Python package, and JSON Python package. 

(4) `src/preprocessing/Dockerfile` - This dockerfile starts with `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - enter the below commands in the CLI:
- cd validation
- chmod a+x docker-shell.sh
- ./docker-shell.sh

**Cross validation, Data Versioning**
- This container reads preprocessed dataset and creates validation split and uses dvc for versioning.
- Input to this container is source GCS location, parameters if any, secrets needed - via docker
- Output is flat file with cross validation splits
  
(1) `src/validation/cv_val.py` - Here we split the FudanSELab data set. Since the data set is not huge, we decided to keep 90% for training and 10% for validation. Our metrics will be monitored on this 10% validation set.

(2) `src/validation/cv_val_sql.py` - Here we split the SQL Create Context data set. Since the data set is not huge, we decided to keep 90% for training and 10% for validation. Our metrics will be monitored on this 10% validation set.

(2) `requirements.txt` - We used following packages to help us with cross validation here: 
- sklearn.model_selection (specifically train_test_split)
- gogle.cloud (specifically storage)
- io (specifically StringIO)
- json

(3) `src/validation/Dockerfile` - This dockerfile starts with `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - enter the below commands in the CLI:
- cd validation
- chmod a+x docker-shell.sh
- ./docker-shell.sh

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 

----
You may adjust this template as appropriate for your project.
