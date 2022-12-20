## Capstone Project - Azure Machine Learning Engineer (Heart Failure Prediction)
For this project, we will create two models: one using Automated Machine Learning (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both models and deploy the most optimal performing model. The model can be consumed from the generated REST endpoint.

![image](https://user-images.githubusercontent.com/64579075/207234565-97be73a1-62f0-46fb-a20e-c94565e89043.png)

## Project Set Up and Installation
1.	This project requires the creation of a compute instance in order to run a Jupyter Notebook & compute cluster to run the Machine Learning Experiments.
2.	We will use an external dataset to train the model to classify heart failure based on clinical records.
3.	Two experiments were run using AutoML & HyperDrive.

            •	automl.ipynb: for the AutoML experiment

            •	hyperparameter_tuning.ipynb: for the HyperDrive experiment.


4.	The Best Model Run, Metrics and a deployed model which consumes a RESTful API Webservice in order to interact with the deployed model.

## Dataset

### Overview
The dataset we will be using in this project is called Heart failure clinical records Data Set and is publicly available from UCI Machine Learning Repository.

The dataset contains medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Here is a [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv) to the data.

### Task
I am using this data in order to predict the DEATH_EVENT i.e. whether or not the patient deceased during the follow-up period (boolean). 

The data includes the following features:

•	age: Age of the patient (years)

•	anaemia: Decrease of red blood cells or hemoglobin (boolean) 

•	creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)

•	diabetes: If the patient has diabetes (boolean)

•	ejection_fraction: Percentage of blood leaving the heart at each contraction

•	high_blood_pressure: If the patient has hypertension (boolean)

•	platelets: Platelets in the blood (kiloplatelets/mL)

•	serum_creatinine: Level of serum creatinine in the blood (mg/dL)

•	serum_sodium: Level of serum sodium in the blood (mEq/L)

•	sex: Gender (woman or man) (binary)

•	smoking: If the patient smokes or not (boolean)

•	time: follow-up period (days)

•	DEATH_EVENT [ Target]: Target column which tells if the patient deceased during the follow-up period


### Access
We downloaded the dataset from the UCI Machine Learning Repository and uploaded it to this GitHub repository. In both notebooks the dataset was read in using Dataset.Tabular.from_delimited_files using the url of that dataset at the UCI machine learning repository site and then registered in Azure if it hadn't been already.

![dataset - register](https://user-images.githubusercontent.com/64579075/208568526-33fe8ce0-a678-494c-994b-607d2f110333.PNG)

![dataset - sample](https://user-images.githubusercontent.com/64579075/208568542-d821c48b-9db4-4c1e-bcc0-9931973eca83.PNG)


## Automated ML
In this part of the project, we make a use of Microsoft Azure Cloud to configure a cloud-based machine learning model and consequently deploy it. We first create a compute target and then train a set of machine learning models leveraging AutoML to automatically train and tune a them using given target metric. 

In this case we selected target metric is “accuracy”. A datastore retrieved by data_store = ws.get_default_datastore() is used to upload the dataset used to train the ML model and it is registered by using the following command

# Create AML Dataset and register it into Workspace
dataset = Dataset.Tabular.from_delimited_files(data_url)
    #Register Dataset in Workspace
    dataset = dataset.register(workspace = ws,name = key,description = description_text)

## AutoML Configuration and Settings

Here we are dealing with a binary classification. Therefore, the argument task is set to “classification” and our target column is “DEATH_EVENT” also need to set label_column_name="DEATH_EVENT". 

The dataset itself specified in training_data=dataset and the compute target that we provisioned is set with compute_target=compute_target.

Besides other arguments that are self-explanatory, to automate Feature engineering AzureML enables this through featurization that needs to be set to True. This way features that best characterize the patterns in the data are selected to create predictive models.

Here is the code to set and configure the AutoML experiment

![Automl-setting-image](https://user-images.githubusercontent.com/64579075/208565775-b7c58225-ab18-4fd2-9ff7-0fab95350992.PNG)

Once the AutoML experiment is completed, we then select the best model in terms of “accuracy” out of all models trained and deploy it using Azure Container Instance (ACI). So that the model can then be consumed via a REST API.

### Results
The AutoML experiment run generated VotingEnsemble algorithm as the best model with accuracy of 0.86617

![Automl-1](https://user-images.githubusercontent.com/64579075/208567896-6e98280e-d517-43ce-bb1c-922e8750c505.PNG)

## Run details

![Automl - rundetails](https://user-images.githubusercontent.com/64579075/208568030-6bfdc920-27be-450d-91fa-6d08b5ad8aec.PNG)

For more comprehensive details please see automl.ipynb notebook.

## Best model in Azure Portal

![Automl - 2](https://user-images.githubusercontent.com/64579075/208568391-ff0e451a-ae4a-4644-ac43-260563e4afd7.PNG)

![Automl - 3](https://user-images.githubusercontent.com/64579075/208568405-28f3dc56-5550-4532-9ed7-e289bb7c8088.PNG)

![Automl - 4](https://user-images.githubusercontent.com/64579075/208568421-d5263024-6f02-4ccd-836e-441a66fb365c.PNG)

![Automl - other metrics](https://user-images.githubusercontent.com/64579075/208568440-6243053d-1ad8-4b04-9f77-aefe95592cf1.PNG)


## Hyperparameter Tuning

The classification algorithm used here is Logistic Regression.Logistic regression is a well-known method in statistics that is used to predict the probability of an outcome, and is especially popular for classification tasks. The algorithm predicts the probability of occurrence of an event by fitting data to a logistic function. Then the training(train.py) script is passed to estimator and HyperDrive configurations to predict the best model and accuracy.

In this section I have used a custom-coded model — a standard Scikit-learn Logistic Regression - which hyperparameters I optimized using HyperDrive.
A Hyperdrive run is used to sweep over model parameters. The following steps are part of the process:




•	Data preprocessing

•	Splitting data into train and test sets

•	Setting logistic regression parameters:

           o	--C - Inverse of regularization strength
           
           o	--max_iter - Maximum number of iterations convergence
           
•	Azure Cloud resources configuration

•	Creating a HyperDrive configuration using the estimator, hyperparameter sampler, and policy

•	Retrieve the best run and save the model from that run


## RandomParameterSampling
Defines random sampling over a hyperparameter search space. In this sampling algorithm, parameter values are chosen from a set of discrete values or a distribution over a continuous range. This has an advantage against GridSearch method that runs all combinations of parameters and requires large amount of time to run.
For the Inverse of regularization strenght parameter I have chosen uniform distribution with min=0.0001 and max=1.0 For the Maximum number of iterations convergence I inputed a range of values (5, 25, 50, 100, 200, 500, 1000)

Parameter sampler

I specified the parameter sampler as such:

param_sampling = RandomParameterSampling(
    {
        "--C": uniform(0.0001, 1.0),
        "--max_iter": choice(5, 25, 50, 100, 200, 500, 1000)
    }
)

## BanditPolicy
Class Defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. This greatly helps to ensure if model with given parameters is not performing well, it is turned down instead of running it for any longer.


    early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
    

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The best model given by HyperDrive resulted in training accuracy of 0.7833333333333333%. The hyperparameters of the model are as follows:

--C = 0.6999143187991206
--max_iter = 500
The best model parameters are retrieved by using this code:

Best Experiment Run:
 Best Run Id: HD_f36a8c91-66f4-4a85-89f8-5feeb35730ae_3
 Accuracy: 0.7833333333333333
 Regularization Strength: 0.6999143187991206
 Max iterations: 500

Improvement

One way to improve the result could be to change the range of hyperparameters to extend the search space.
Other ways include changing the ML model completely or use a data set with much more data records if that would be a posibility

## Screenshots

![HPHD - 1](https://user-images.githubusercontent.com/64579075/208570636-3b570a3f-3408-4b45-96ce-9028878ea672.PNG)

![HPHD - 2](https://user-images.githubusercontent.com/64579075/208570655-6646f8fd-aaa9-48ea-8fca-58029ad4b01d.PNG)

![HPHD - 3](https://user-images.githubusercontent.com/64579075/208570669-70235713-04bf-4f57-8bbe-bc81450882cf.PNG)

![HPHD - 4](https://user-images.githubusercontent.com/64579075/208570680-0cabb2af-0478-411a-8414-b62e871dc21d.PNG)


## Model Deployment
The deployment is done following the steps below:

Selection of an already registered model

Preparation of an inference configuration

Preparation of an entry script

Choosing a compute target

Deployment of the model

Testing the resulting web service

   HyperDrive’s best run accuracy = 78.33%

   AutoML’s best run accuracy = 86.61%

Thus,Automl's model has the highest accuracy.The model with the best accuracy is deployped,so the AutoML's best model is deployed.

Initially, the best model is registered and it's necessary files are downloaded.Then the Environment and inference is created with the help of required conda dependencies and score.py script file which has the intialization and exit function defined for the best model and the model is deployed with ACI(Azure Container Instance) and configurations such as cpu_cores=1, memory_gb=1.Once the deployment is sucessful, applications insights is enabled and the state of the service is verified.Then the behaviour of the endpoint is analyzed and the service is deleted

![MD - setting](https://user-images.githubusercontent.com/64579075/208571869-3210cfa0-5a9d-48af-a064-da4ec955615c.PNG)

![MD -1](https://user-images.githubusercontent.com/64579075/208571735-e93a2799-5bc7-4615-9e22-6fc8162a0db3.PNG)

![MD - 2](https://user-images.githubusercontent.com/64579075/208571743-1c464e43-c940-461f-9bb9-f9c072d80d19.PNG)

![keys](https://user-images.githubusercontent.com/64579075/208571994-b3b0243b-32d3-451e-a8c0-d40a7e5039ca.PNG)

![MD - 3](https://user-images.githubusercontent.com/64579075/208571757-8ce67e44-d61d-4676-84cc-a63be81a91cc.PNG)

![Endpoint consume](https://user-images.githubusercontent.com/64579075/208572244-ac945bce-59f3-4af4-9b08-be78b8b13216.PNG)

![MD - 4](https://user-images.githubusercontent.com/64579075/208571766-acccb144-4b93-4dd9-8b36-f063c7ae056c.PNG)


## Screen Recording
Here is the screencast[link] https://drive.google.com/file/d/13_FM60ZxVswgFP0zMaoQZL2b4EwsLB4x/view?usp=sharing shows the entire process of the workflow:

*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

The screen recording can be found here and it shows the project in action.
