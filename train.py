from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def preprocess_data(data_df):
    
    y_df = data_df.pop("DEATH_EVENT")
    x_df = data_df
    
    return x_df, y_df

# Create TabularDataset using TabularDatasetFactory
url_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
dataset = TabularDatasetFactory.from_delimited_files(path=url_path)
data_df = dataset.to_pandas_dataframe()

# Preview of the first five rows
data_df.head()

# Explore data
data_df.describe()



# Split data to features and labels dataframe
x_df, y_df = preprocess_data(data_df)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':