*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include. https://github.com/Harini-Pavithra/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree

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

# Run details

![Automl - rundetails](https://user-images.githubusercontent.com/64579075/208568030-6bfdc920-27be-450d-91fa-6d08b5ad8aec.PNG)

For more comprehensive details please see automl.ipynb notebook.

## Best model in Azure Portal

![Automl - 2](https://user-images.githubusercontent.com/64579075/208568391-ff0e451a-ae4a-4644-ac43-260563e4afd7.PNG)

![Automl - 3](https://user-images.githubusercontent.com/64579075/208568405-28f3dc56-5550-4532-9ed7-e289bb7c8088.PNG)

![Automl - 4](https://user-images.githubusercontent.com/64579075/208568421-d5263024-6f02-4ccd-836e-441a66fb365c.PNG)

![Automl - other metrics](https://user-images.githubusercontent.com/64579075/208568440-6243053d-1ad8-4b04-9f77-aefe95592cf1.PNG)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

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



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
