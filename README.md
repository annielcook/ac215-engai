Team EngAi Milestone 6 Deliverable
==============================

AC215 - Milestone 6

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

#### Kubernetes Deployment ####
We deployed the frontend and backend containers to the kubernetes cluster to handle key distributed systems issues such as load balancing and failover.
Note that the cluster contains more than 1 node to handle peak traffic loads. The ansible playbooks you can find allow us
to manage code as infrastructure. As can be seen in the CI/CD section below these ansible scripts are called upon
any changes to the code, so that new changes can be deployed to production rapidly and continuously. 


Here is a screenshot of the Kubernetes cluster we are running in GCP:
![image](https://github.com/annielcook/ac215-engai/blob/main/images/kubernetes-multi-node-screenshot.png)

#### Code Structure ####
The following are the folders from the previous milestones: 

- api-service
- deployment
- dvc
- frontend-react
- model-deployment
- models
- preprocessing
- tensorizing
- validation
- workflow

**API Service Container** 
This container has all the python files to run and expose the backend apis. These APIs
are called by the client side code to actually run model inference on the server side.

To run the container locally:

- Open a terminal and go to the location  src/api-service
- Here you'll find docker-shell.sh. Run sh docker-shell.sh
- Once inside the docker container run uvicorn_server
- To view and test that the APIs are up. Go to http://localhost:9000/docs


**Frontend Container** 

This container contains all the files used for the react app. Note that this folder contains
docker files for both development and production. To run the container locally:

- Open a terminal and go to the location src/frontend
- Run sh docker-shell.sh 
- Once inside the docker container please run yarn start (if you haven't already run yarn install once)
- Go to http://localhost:3000 to access the app locally

**Deployment Container** 

This container manages building and deploying each of our application containers. All of the containers get deployed to 
GCR and the service starts to run in GCP

To run the container locally:

- Open a terminal and go to src/deployment
- Run sh docker-shell.sh
- Build and Push Docker Containers to GCR (Google Container Registry) by running:

ansible-playbook deploy-docker-images.yml -i inventory.yml

- Create and deploy the cluster using:

ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present

- Note the nginx_ingress_ip that was outputted by the cluster command 
- Visit http:// YOUR INGRESS IP.sslip.io to view the website

If you want to run our ML flows checkout the directions under workflow. The following are the commands to run the pipelines:

- Go to src/workflow
- Run sh docker-shell.sh 
- Run python3 workflow.py to get a list of directions on how to run each stage of the pipeline 

#### CI / CD: Deploy Using Github Actions ####
We added CI/CD using GitHub Actions, such that we can trigger deployment or any other partial or full pipeline using GitHub Events. Our yaml file which instantiates the actions and associates them with events can be found under .github/workflows/ci-cd.yml.

Our CI/CD workflow accomplishes a few things
- App deployment: utilizing the existing deployment container, on code changes we kick off the flow of building the docker image, pushing it to GCR, deploying changed containers to update the k8s cluster with ansible, and re-running vertex AI jobs if needed
- Triggering individual steps: using different commit messages ("/run-data-preprocessing", "/run-ml-pipeline", etc), we can trigger individual steps like data pre-processing or running the full ml pipeline. Each of these steps is executed through a cli command from `cli.py`. 

Below is a screenshot from the Github actions dashboard showing a few successful runs. First we ran it on a separate `gha-test` branch and then once it was up and working, we merged it into `main` and ran there.
<img width="1082" alt="image" src="https://github.com/annielcook/ac215-engai/assets/6455793/e54b237d-cdaa-457a-90cc-a018a2d83e2e">



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

#### Application Design ####

You can find the Solutions Architecture and Technical Architecture diagrams below. The two diagrams detail how the various components of the system work together to classify dog images. 

**Solution Architecture**
![image](https://github.com/annielcook/ac215-engai/blob/main/images/solution-arch.png)

**Technical Architecture**
![image](https://github.com/annielcook/ac215-engai/assets/1981839/2eff57ac-42d0-4b46-855b-286a0fa9f646)

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

Similar to the ConNextBase architecture, this model did not generalize well and overfit the training data, achieving a max training accuracy of 87.7% and max validation accuracy of 56.3%

![Screenshot 2023-11-20 at 1 15 13 PM](https://github.com/annielcook/ac215-engai/assets/48300750/44c825a7-4f9c-4098-8504-f52744f4e61d)

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


#### API Service
<img width="748" alt="Screenshot 2023-11-20 at 5 38 34 PM" src="https://github.com/annielcook/ac215-engai/assets/22153363/244b7316-9693-4234-b43c-4e61d3248236">
The `api-service` provides two endpoints, the index and the predict endpoints. The `/predict` endpoint is called from the frontend with an image to make a model inference. 

The `ModelController` is responsible for calling either the local model (saved in the container) or the remote model (stored on VertexAI)

#### Front-End Development

##### Components
We have three components in the [Components](https://github.com/annielcook/ac215-engai/tree/main/src/frontend-react/src/app/components) directory.

[Footer](https://github.com/annielcook/ac215-engai/tree/main/src/frontend-react/src/app/components/Footer) contains the footer that stores the history of the past 5 search results (just the predicted breed, not the probabilities).

[ImageUpload](https://github.com/annielcook/ac215-engai/tree/main/src/frontend-react/src/app/components/ImageUpload) contains the interface for uploading an image to the website, making a call to the model (depending on ModelToggle), returning the predicted breed and confidence level (probability), and storing that predicted breed in the Footer as part of the search history.

[ModelToggle](https://github.com/annielcook/ac215-engai/tree/main/src/frontend-react/src/app/components/ModelToggle) has a dropdown for the user to select either our Hosted or Local Model. We included both to show the difference in response times. The model itself is the same so the performance in terms of accuracy is expected to be the same as well. The parameter is passed from the user-selected dropdown as part of the formData argument that is read in DataService in the services section (see below).

##### Services
We have two React files in the [Services](https://github.com/annielcook/ac215-engai/tree/main/src/frontend-react/src/app/services) directory.

[BreedParse](https://github.com/annielcook/ac215-engai/blob/main/src/frontend-react/src/app/services/BreedParse.js) is used to extract the reader friendly version of the predicted breed species name to display it in the results section of ImageUpload and append it to the history of the past 5 results in the Footer.

[DataService](https://github.com/annielcook/ac215-engai/blob/main/src/frontend-react/src/app/services/DataService.js) is used to make the server call to the API endpoint to select the right model, depending on the selection in the ModelToggle component.


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
