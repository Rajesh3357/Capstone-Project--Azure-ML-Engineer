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

## Automated ML
Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


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
