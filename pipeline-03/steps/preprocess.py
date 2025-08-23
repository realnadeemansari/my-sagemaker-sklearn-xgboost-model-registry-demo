import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import mlflow
import traceback
import threading
import time

from utils import start_mlflow_ui, upload_to_s3


def preprocess(input_data_s3_uri: str, experiment_name="main_experiment", run_name="run-01") -> tuple:
    try:
        # run in background thread
        print("[DEBUG] Starting MLflow UI for preprocessing")
        threading.Thread(target=start_mlflow_ui, daemon=True).start()
        time.sleep(60)
        print("[DEBUG] Starting preprocessing data...")        
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(run)
            with mlflow.start_run(run_name="DataPreprocessing", nested=True):
                mlflow.autolog()
                columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
                cat_columns = ['Type']
                num_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                target_column = 'Machine failure'
                print("[DUBUG] input data path", input_data_s3_uri)
                df = pd.read_csv(input_data_s3_uri)
                print("[DEBUG] Data loaded")
                df = df[columns]

                training_ratio = 0.8
                validation_ratio = 0.1
                test_ratio = 0.1
            
                X = df.drop(target_column, axis=1)
                y = df[target_column]
            
                print(f'Splitting data training ({training_ratio}), validation ({validation_ratio}), and test ({test_ratio}) sets ')
                mlflow.log_param("training_ratio",training_ratio)
                mlflow.log_param("test_ratio",training_ratio)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0, stratify=y)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_ratio/(validation_ratio+training_ratio), random_state=2, stratify=y_train)
            
                # Apply transformations
                transformer = ColumnTransformer(transformers=[('numeric', StandardScaler(), num_columns),
                                                            ('categorical', OneHotEncoder(), cat_columns)],
                                                remainder='passthrough')
                featurizer_model = transformer.fit(X_train)
                X_train = featurizer_model.transform(X_train)
                X_val = featurizer_model.transform(X_val)
            
                print(f'Shape of train features after preprocessing: {X_train.shape}')
                print(f'Shape of validation features after preprocessing: {X_val.shape}')
                print(f'Shape of test features after preprocessing: {X_test.shape}')
                
                y_train = y_train.values.reshape(-1)
                y_val = y_val.values.reshape(-1)
                
                print(f'Shape of train labels after preprocessing: {y_train.shape}')
                print(f'Shape of validation labels after preprocessing: {y_val.shape}')
                print(f'Shape of test labels after preprocessing: {y_test.shape}')
                print("[DEBUG] Data preprocessed")
                model_file_path="./opt/ml/model/sklearn_model.joblib"
                os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
                joblib.dump(featurizer_model, model_file_path)
                print("[DEBUG] Model saved")
        upload_to_s3("./opt/ml/model", "sagemaker07272025", "mySavedModels/")
        upload_to_s3("mlruns", "sagemaker07272025", "mlruns/")
        return X_train, y_train, X_val, y_val, X_test, y_test, featurizer_model, run_id
    except Exception as e:
        err_msg = '\n' + ''.join(traceback.format_tb(e.__traceback__))
        print("Error", err_msg)
        raise e