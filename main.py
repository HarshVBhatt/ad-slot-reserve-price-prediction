from components import data_ingestion, feature_eng, model, logger, output_transformations as tfr
from pyspark.sql import SparkSession
import os
import warnings
warnings.filterwarnings('ignore') 

spark = SparkSession.builder \
    .appName("DataIngestion") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def IngestData():
    bucket = "reserve-price-prediction"
    key = "rawdata/Dataset.csv"

    df = data_ingestion.read_from_s3(bucket = bucket, key = key)
    data_ingestion.get_metadata(data = df)

    return df

def TransformFeatures(df):
    return feature_eng.feature_engineering(df)

def main():
    data = IngestData()
    transformed_data = TransformFeatures(df = data)
    train_df, test_df = model.split_train_test(data = transformed_data, date = "2019-06-23")
    X_train, y_train = model.split_X_y(data = train_df)
    X_test, y_test = model.split_X_y(data=test_df)

    xgb_model, parameters = model.train_model(X_train=X_train, y_train=y_train, n_estimators = 250, random_state = 42)
    predictions = model.get_predictions(model=xgb_model, X_test=X_test)

    model_path, timestamp = logger.log_run(model = model, parameters = parameters, y_pred=predictions, y_test=y_test)
    logger.save_to_wandb(model_path = model_path, parameters = parameters)

    predictions_inv = tfr.inv_transform_target(orig_df=transformed_data, predictions = predictions)
    output_transformed_df = tfr.add_predictions(test_df = test_df, predictions = predictions_inv)
    final_df = tfr.get_original_columns(original_df = data, new_df = output_transformed_df)
    logger.save_csv_locally(object=final_df, path = f"artifacts/model_{timestamp}/full_output.csv")

    artifacts_path = "/".join(list(model_path.split("/"))[:-1])
    model_id = model_path.split("/")[1]
    bucket = "reserve-price-prediction"
    for file in os.listdir(artifacts_path):
        key = f"{artifacts_path}/{file}"
        logger.upload_to_s3(file_path = key, bucket_name = bucket, object_name = key)
    logger.upload_as_parquet(object = final_df, bucket_name = bucket, object_path = f"output_files/{model_id}")

if __name__ == "__main__":
    main()