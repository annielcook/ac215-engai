Team EngAi Milestone 5 Deliverable
==============================

AC215 - Milestone 5

Project Organization
```bash
├── LICENSE
├── notebooks
│   ├── breed_labels.txt
│   ├── DogNet_Breed_Distillation.ipynb
│   ├── ExploratoryDataAnalysis.ipynb
│   └── model_testing.ipynb
├── README.md
├── requirements.txt
└── src
    ├── api-service
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── api
    │   │   ├── model.py
    │   │   └── service.py
    │   ├── config
    │   │   ├── breed-to-index.json
    │   │   ├── index-to-breed.json
    │   │   ├── model-controller-config.json
    │   │   └── util.py
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   └── secrets
    │       └── wandb.json
    ├── deployment
    │   ├── Dockerfile
    │   ├── deploy-create-instance.yml
    │   ├── deploy-docker-images.yml
    │   ├── deploy-provision-instance.yml
    │   ├── deploy-setup-containers.yml
    │   ├── deploy-setup-webserver.yml
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── inventory.yml
    │   ├── loginProfile
    │   ├── nginx-conf
    │   │   └── nginx
    │   │       └── nginx.conf
    │   └── secrets
    │       ├── deployment.json
    │       ├── gcp-service.json
    │       ├── ssh-key-deployment
    │       └── ssh-key-deployment.pub
    ├── dvc
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   └── team-engai-dogs.dvc
    ├── frontend-react
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── Dockerfile.dev
    │   ├── package.json
    │   ├── public
    │   │   ├── favicon.ico
    │   │   ├── index.html
    │   │   └── manifest.json
    │   ├── src
    │   │   ├── app
    │   │   │   ├── App.css
    │   │   │   ├── App.js
    │   │   │   ├── background.png
    │   │   │   ├── components
    │   │   │   │   ├── Footer
    │   │   │   │   │   ├── Footer.css
    │   │   │   │   │   └── Footer.js
    │   │   │   │   ├── ImageUpload
    │   │   │   │   │   ├── ImageUpload.css
    │   │   │   │   │   └── ImageUpload.js
    │   │   │   │   └── ModelToggle
    │   │   │   │       ├── ModelToggle.css
    │   │   │   │       └── ModelToggle.js
    │   │   │   └── services
    │   │   │       ├── BreedParse.js
    │   │   │       └── DataService.js
    │   │   └── index.js
    │   └── yarn.lock
    ├── model-deployment
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cli.py
    │   ├── docker-entrypoint.sh
    │   └── docker-shell.sh
    ├── models
    │   └── resnet152v2
    │       ├── Dockerfile
    │       ├── Pipfile
    │       ├── Pipfile.lock
    │       ├── distiller.py
    │       ├── docker-shell.sh
    │       ├── dog_breed_dataset
    │       │   └── images
    │       │       └── Images
    │       ├── model_training_age_dataset.py
    │       ├── model_training_breed_dataset.py
    │       ├── model_training_breed_dataset_distillation.py
    │       ├── model_training_breed_dataset_pruned.py
    │       ├── run-model.sh
    │       ├── secrets
    │       │   └── data-service-account.json
    │       └── util.py
    ├── preprocessing
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── ResizeDogImages.ipynb
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── preprocess_age.py
    │   ├── preprocess_breed.py
    │   └── util.py
    ├── pwd
    ├── secrets
    │   ├── data-service-account.json
    │   └── wandb.json
    ├── tensorizing
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── curr_image
    │   ├── curr_image.jpg
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── hold_working_age.py
    │   ├── secrets
    │   │   └── data-service-account.json
    │   ├── tensorize_age_dataset.py
    │   └── tensorize_breed_dataset.py
    ├── validation
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cv_val.py
    │   ├── cv_val_sql.py
    │   ├── docker-shell.sh
    │   └── requirements.txt
    └── workflow
        ├── Dockerfile
        ├── Pipfile
        ├── Pipfile.lock
        ├── age_model_training.yaml
        ├── cli.py
        ├── data_preprocessing.yaml
        ├── docker-entrypoint.sh
        ├── docker-shell.sh
        ├── pipeline.yaml
        ├── secrets
        │   └── compute-service-account.json
        └── tensorizing.yaml
```

32 directories, 109 files


# AC215 - Milestone 5 - DogWatcher (powered by DogNet)

**Team Members**
Nevil George, Juan Pablo Heusser, Curren Iyer, Annie Landefeld, Abhijit Pujare

**Group Name**
EngAi Group

**Project**
In this project, we aim to build an application that can predict a dog's breed and age using a photo.  

### Milestone 5 ###

In this milestone we worked on multiple aspects of the project:

      (1) Deployment of the web service to GCP [/src/deployment/](src/deployment/)
      
      (2) Frontend/React container [/src/frontend-react/](src/frontend-react/)
      
      (3) API service [/src/api-service/](src/api-service/)

      (4) Add model deployment to Vertex AI [/src/model-deployment/](src/model-deployment/)

      (5) Switching from Model Pruning to Knowledge Distillation as compression technique

#### Deployment Strategy ####

We used Ansible to automate the provisioning and deployment of our frontend and backend containers to GCP. Below you can find a screenshot of the VM that's running our service on GCP.  

![image](https://github.com/annielcook/ac215-engai/assets/1981839/4ee0bf8d-4467-405a-ba66-ce5a847649d5)

Additionally, you can find a screenshot that shows the container images we have pused to the GCP container repository:

![image](https://github.com/annielcook/ac215-engai/assets/1981839/a370ecfd-0201-4eb0-986e-59542450d601)


**Deployment Container**
[/src/deployment/](src/deployment/)

This container builds the containers, creates and provisions a GCP instance and then deploys those containers to those intances.

If you wish to run the container locally :

* Navigate to  src/deployment in your terminal
* Run sh docker-shell.sh
* Build and Push Docker Containers to GCR (Google Container Registry) by running the following yaml"

`ansible-playbook deploy-docker-images.yml -i inventory.yml`

* Create Compute Instance (VM) Server which will host the containers

`ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present`

* Provision Compute Instance in GCP to setup all required software

`ansible-playbook deploy-provision-instance.yml -i inventory.yml`

* Install Docker Containers on the Compute Instance

`ansible-playbook deploy-setup-containers.yml -i inventory.yml`

* Setup Webserver on the Instance

`ansible-playbook deploy-setup-webserver.yml -i inventory.yml`

#### Adding Model Deployment to Vertex AI
[/src/model-deployment/](src/model-deployment/)
In order to finish out the model pipeline which powers the ML application, we added the final step of model deployment to the Vertex AI pipeline. This step utilizes a command line interface to take the model from Weights & Biases, upload it to Google Cloud Storage, and deploy it to Vertex AI. With the final step in place, the end to end model development from data processing, to tensorizing, to model training, and now model deployment are all part of a unified pipeline.
![WhatsApp Image 2023-11-16 at 8 49 40 PM](https://github.com/annielcook/ac215-engai/assets/6455793/ffd82444-e3c0-4dba-bbcf-160c310bde07)

To use just the model deployment service, first launch the service with `./docker-shell.sh` to get to the interpreter. 

* Upload the model from Weights & Biases to GCS

`python3 cli.py --upload`

* Deploy the model to Vertex AI
 
 `python3 cli.py --deploy`


#### Model Distillation

 [/notebooks/DogNet_Breed_Distillation.ipynb](/notebooks/DogNet_Breed_Distillation.ipynb)

 In milestone 4 we used model pruning as our compression technique but realized that distillation was more suitable for our application since most of the models layers were not being trained. All of the code used to test different model combinations and distillation can be found in the notebook linked above.

 We tested different base architectures for both the teacher and the student model. 

##### Teacher model:

###### ResNet152v2: Total Parameters - 59,630,968 | Total Size - 227.47 MB 

With this model architecture we obtained a maximum validation accuracy of 82.5% on epoch 20. The model learned fairly quickly compared to other architectures, achieving a 68% validation accuracy on the first epoch.

![Screenshot 2023-11-19 at 11 25 48 PM](https://github.com/annielcook/ac215-engai/assets/48300750/d3dbb014-6309-408c-9f7c-5500b081653d)

###### ConNeXtBase: Total Parameters - 88,353,784 | Total Size - 337.04 MB

This base architecture did not perform well on the dogs dataset, as we only achieved a 42.25% maximum validation accuracy on epoch 27.

![Screenshot 2023-11-20 at 7 02 44 AM](https://github.com/annielcook/ac215-engai/assets/48300750/4f0c1b62-1a63-41b4-b025-0e6569e07095)

###### DenseNet201: Total Parameters - 19,557,304 | Total Size - 74.61 MB

Using the DenseNet201 model architecture we achieved very good results for such a small model, yet it still obtained a lower max validation accuracy compared to ResNet152v2, of 81.9%. The difference is minimal but as a team we decided to use ResNet152v2 as our teacher model.

![Screenshot 2023-11-20 at 7 03 43 AM](https://github.com/annielcook/ac215-engai/assets/48300750/f8435467-61e3-42f5-9044-8ffb1f3dc2e5)



##### Student model:

###### ResNet50: Total Parameters - 24,855,024 | Total Size - 94.81 MB

This model architecture did not perform well on the dataset. The training accuracy was around 84% by the end of the 30 epochs, while the validation accuracy was around just 24% meaning that the model was not generalizing well, and overfitting the training data.

![Screenshot 2023-11-20 at 7 59 13 AM](https://github.com/annielcook/ac215-engai/assets/48300750/d7050a2e-5116-4263-af60-2de4eb3a99e5)

###### ConNextSmall: Total Parameters - 50,076,880 | Total Size - 191.03 MB

Similar to the ConNextBase architecture, this model did not generalize well and overfit the training data, achieving a max training accuracy of XX% and max validation accuracy of XX%

###### DenseNet121: Total Parameters - 7,788,720 | Total Size - 29.71 MB

With this base model architecture we achieved a maximum validation accuracy of 71.6% by epoch 17. The model was able to learn quickly initially and the accuracy obtained was significantly lower than that obtained with the teacher model, making it a prime candidate for model distillation.

![Screenshot 2023-11-20 at 7 09 19 AM](https://github.com/annielcook/ac215-engai/assets/48300750/a44727f4-0a2e-4a17-a313-7cc6f488e266)

###### Model Distillation: Total Parameters - 7,788,720 | Total Size - 29.71 MB

For model distillation we decided to use the teacher model with the ResNet152v2 base architecture and we built a new student model using the DenseNet121 architecture. Then based on the contents reviewed in class we proceeded to implement the distillation training loop and train the student model by distilling from the teacher model. We obtained a 92.6% validation accuracy, even greater than with the teacher model, on epoch 28. Using distillation we managed to compress the teacher model 7.65x and achieve better validation accuracy.

![Screenshot 2023-11-20 at 10 31 21 AM](https://github.com/annielcook/ac215-engai/assets/48300750/bc035e67-d762-427c-9fbc-89454a21a3f0)


This result es extremely positive as the distilled student model achieved a better validation accuracy than the teach model. Even more so, this model obtained a validation accuracy similar to top SOTA models for Fine-Grained Image Classification on the Stanford Dogs dataset. 

(https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1)

The Nº1 model on this list, the ViT-NeT model achieved a 93.6% accuracy on the same dataset. Our results would place our distilled student model in the top 10 of this list.

Below is a comparison table obtained from the ViT-NeT paper. 

![Screenshot 2023-11-20 at 10 24 08 AM](https://github.com/annielcook/ac215-engai/assets/48300750/825a1fc2-d9a3-445d-a0b8-f2944fb42228)

Source: 
Kim, S., Nam, J., & Ko, B. C. (2022). ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder. In Proceedings of the 39th International Conference on Machine Learning (PMLR 162). Baltimore, Maryland, USA.

--------
# AC215 - Milestone 4 - DogWatcher (powered by DogNet)

**Team Members**
Nevil George, Juan Pablo Heusser, Curren Iyer, Annie Landefeld, Abhijit Pujare

**Group Name**
EngAi Group

**Project**
In this project, we aim to build an application that can predict a dog's breed and age using a photo.  

### Milestone 4 ###

In this milestone, we worked on three aspects of the project: 
   (1) Distillation and pruning to optimize the model for size and performance
   (2) Running all the containers in the vertex ai service to get the end to end data pipeline set up. 


**Machine Learning Workflow Using Kubeflow and Vertex**
Under the workflow directory, you can find: 

 - cli.py - this script  calls the vertex ai service to run the preprocessing, tensorizing and model training containers in a sequential data pipeline. Currently, we have used this data pipeline to preprocess, tensorize, and run the models on the kaggle datasets we have collected.
 - .yaml files -  You can see that we used the kubeflow dsl annotations in cli.py to specify the different components of the pipeline that ultimately get composed into a yaml file. The data_preprocessing.yaml file has been included as a reference in this repository.  

Looking ahead, we hope to scrape images of dogs from google images and run them on this data pipeline to refine our models. 

**Model Optimization**
Initially we optimized the model using pruning, with Tensorflow's model optimization library. We used a a polinomial decay sparsity scheduler, to increase the sparsity between epochs, from 0.5 to 0.8. We were able to maintain the same validation accuracy through this method. However, this optimization technique was not optimal for a transfer learning model, because we were only pruning the last two dense layers of the model, while the bulk of parameters was concentrated in ResNet152v2. With this key learning, we implemented model distillation. This involves taking the training from a teacher model (our trained ResNet152v2 model from above) and applying it to a blank slate student model. With distillation, the result is a model orders of magnitude smaller. The accuracy, however, is lower. For the next milestone we will continue to explore optimizations and their tradeoffs in order to determine the ideal decisions for this implementation. 

**Datasets**
Kaggle Stanford Dogs - https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
 - A dataset of dog pictures with their breeds.

Kaggle Dog Age - https://www.kaggle.com/datasets/user164919/the-dogage-dataset
 - A dataset of dog pictures with their ages. 

**Preprocess container**
- This container reads 2.8GB of data, resizes the images (224x224), and stores it back to GCP.
- Input to this container is source and destincation GCS location, secrets needed - via docker
- Output from this container stored at GCS location

_Following the Professor's advice, we didn't use Dask for a dataset of this size for now. We are eager to use it in subsequent changes._

(1) `src/preprocessing/preprocess_age.py`  - Here we preprocess the Stanford Dogs data set. We read in every image file from the source GCS bucket, resize them to the required 224x224 size, then re-upload them to the processed GCS bucket with the label ('Young','Adult','Source') appended to the file name.

(2) `src/preprocessing/preprocess_breed.py`  - Here we preprocess the Dog Age data set. We read in every image file from the source GCS bucket, aresize them to the required 224x224 size, then re-upload them to the processed GCS bucket. The label is already accounted for in the filename.

(3) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - Google Cloud Python package, and JSON Python package. 

(4) `src/preprocessing/Dockerfile` - This dockerfile starts with `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - enter the below commands in the CLI:
- cd validation
- chmod a+x docker-shell.sh
- ./docker-shell.sh

**Cross validation, Data Versioning**

_Given our use of Tensorflow, this section may no longer be needed._

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

**Tensorizing**
Given that our dataset had 1000s of images we decided to tensorize the data and store it in GCP. Tensorizing the data involved downloading all of the resized data and then serializing it. To serialize the image data we used the raw image bytes, height, width, channel and label and then serialized it using the tf.train.Example call. We then stored each of these tensorized records back to GCP in the same directory structure as the resized data directory structure. 

**Notebooks** 

This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
We added the following files:
 - ExploratoryDataAnalysis.ipynb -- used to explore our datasets, understand the labels involved, and the count of samples for each label.
![Age Label piechart](https://github.com/juanpheusser/ac215_engai/assets/22153363/7828c6f6-2459-4834-a699-01b0f7788345)
This analysis informed us that dog age model might be biased towards younger dogs because we have more data points for them. 

 - model_testing.ipynb -- used to run 3 different models to check which one performed best (informing our choice of model). We used transfer learning with 2 base architectures (ResNet152V2 and InceptionV3).

**Notebooks** 

This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
We added the following files:
 - ExploratoryDataAnalysis.ipynb -- used to explore our datasets, understand the labels involved, and the count of samples for each label.
![Age Label piechart](https://github.com/juanpheusser/ac215_engai/assets/22153363/7828c6f6-2459-4834-a699-01b0f7788345)
This analysis informed us that dog age model might be biased towards younger dogs because we have more data points for them. 

 - model_testing.ipynb -- used to run 3 different models to check which one performed best (informing our choice of model). We used transfer learning with 2 base architectures (ResNet152V2 and InceptionV3).

**GCP Bucket Structure**

------------
     team-engai-dogs
      ├── dog_age_dataset/
            ├── Expert_Train/
            ├── PetFinder_All/
      ├── dog_breed_dataset/
            ├── annotations/
            ├── images/
      └── dvc_store

--------
We have the same structure for the tensorized data as well, in bucket `team-engai-dogs-tensorized`.


----
