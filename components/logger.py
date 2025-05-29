import wandb
import joblib
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
from model import score


def get_timestamp():
    date = "_".join(str(datetime.now().date()).split("-"))
    time = "_".join(str(datetime.now().time())[:8].split(":"))
    timestamp = date + "_" + time
    return timestamp

def log_run(model, parameters, y_pred, y_test):
    timestamp = get_timestamp()
    base_path = f"artifacts/model_{timestamp}"
    os.mkdir(base_path)
    
    model_path = f"{base_path}/model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    parameters_path = f"{base_path}/parameters.json"
    rmse = score(y_true=y_test, y_pred=y_pred)
    parameters["RMSE"] = rmse
    with open(parameters_path, "w") as f:
        json.dump(parameters, f)
    print(f"Parameters saved to {parameters_path}")

    predictions_path = f"{base_path}/predictions.csv"
    predictions_df = pd.DataFrame({"Actual": y_test.reshape(-1), "Predictions": y_pred})
    predictions_df.to_csv(predictions_path, index = False)
    print(f"Predictions saved to {predictions_path}")

    return model_path, timestamp

def save_to_wandb(model_path, parameters):
    run = wandb.init(project = "reserve_price_prediction")

    artifact = wandb.Artifact("xgbregressor_model", type = "model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print("Model logged to WandB succesfully")

    for k,v in parameters.items():
        run.log({k:v})
        print(f"{k} logged to WandB")
    print("All parameters logged succesfully")

    run.finish()

def save_csv_locally(object, path):
    try:
        object.to_csv(path, index = False)
        return
    except Exception as e:
        return e
    
def upload_to_s3(s3, file_path, bucket_name, object_name):    
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading {file_path} to s3://{bucket_name}/{object_name}: {e}")

def upload_as_parquet(object, bucket_name, object_path):
    try:
        object.write.mode("overwrite").parquet(f"s3a://{bucket_name}/{object_path}/")
        print(f"Successfully uploaded {object} to s3://{bucket_name}/{object_path}/")
    except Exception as e:
        print(f"Error uploading {object} to s3://{bucket_name}/{object_path}: {e}")
