Team EngAi Milestone 2 Deliverable
==============================

# AC215 - Milestone2 - mApp

**Team Members**
Nevil George, Juan Pablo Heusser, Curren Iyer, Annie Landefeld, Abhijit Pujare

**Group Name**
EngAi Group

**Project**
In this project we aim to build an application that can generate SQL Data Schemas and the corresponding Python Classes based on text prompt describing the application for which the data schemas are going to be used.

### Milestone2 ###

We gathered 2 datasets from the following sources:

FudanSELab - https://github.com/FudanSELab/ClassEval
 - Description of 100 python classes and 412 methods, along with their corresponding code. - Total Size: 2.2 MB

SQL Create Context - https://huggingface.co/datasets/b-mc2/sql-create-context
 - Collection of 88,577 hand annotated SQL queries and SQL Create Table statements. Each data point is comprised of a short description of what the SQL statement does, a SQL statement and a SQL Create Table statement for context- 20.7 MB


**Preprocess container**
- This container reads 22.9MB of data, deletes fields that are not relevant to our application, and stores it back to GCP.
- Input to this container is source and destincation GCS location, parameters for deleting fields, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we preprocess both the SQL Create Context dataset and the FudanSELab dataset. We delete 
=======
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
                  ├── cv_dvc.py
                  └── requirements.txt


--------
# AC215 - Milestone2 - ButterFlyer

**Team Members**
Pavlov Protovief, Paolo Primopadre and Pablo El Padron

**Group Name**
Awesome Group

**Project**
In this project we aim to develop an application that can identify various species of butterflies in the wild using computer vision and offer educational content through a chatbot interface.

### Milestone2 ###

We gathered dataset of 1M butterflies representing 17K species. Our dataset comes from following sources - (1),(2),(3) with approx 100GB in size. We parked our dataset in a private Google Cloud Bucket. 

**Preprocess container**
- This container reads 100GB of data and resizes the image sizes and stores it back to GCP
- Input to this container is source and destincation GCS location, parameters for resizing, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we do preprocessing on our dataset of 100GB, we reduce the image sizes (a parameter that can be changed later) to 128x128 for faster iteration with our process. Now we have dataset at 10GB and saved on GCS. 

(2) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `special butterfly package` 

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Cross validation, Data Versioning**
- This container reads preprocessed dataset and creates validation split and uses dvc for versioning.
- Input to this container is source GCS location, parameters if any, secrets needed - via docker
- Output is flat file with cross validation splits
  
(1) `src/validation/cv_val.py` - Since our dataset is quite large we decided to stratify based on species and kept 80% for training and 20% for validation. Our metrics will be monitored on this 20% validation set. 

(2) `requirements.txt` - We used following packages to help us with cross validation here - `iterative-stratification` 

(3) `src/validation/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 

For Milestone 2 - See branch `milestone2`
=======
----
You may adjust this template as appropriate for your project.
