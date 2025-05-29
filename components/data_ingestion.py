from pyspark.sql import SparkSession         

spark = SparkSession.builder \
    .appName("DataIngestion") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
    .getOrCreate()

bucket = "reserve-price-prediction"
key = "rawdata/Dataset.csv"

def read_from_s3(bucket = bucket, key = key):
    df = spark.read.csv(f"s3a://reserve-price-prediction/rawdata/Dataset.csv", header = True, inferSchema=True)
    return df

def get_metadata(data):
    print(f"Number of rows: {data.count()}")
    print(f"Number of columns: {len(data.columns)}")
    print(f"Columns:\n{data.columns}")