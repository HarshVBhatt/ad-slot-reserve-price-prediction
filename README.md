# End-to-end Ad Slot Reserve Price (Base CPM) Prediction System
End-to-end MLOps project to build a ML system to predict Ad slot reserve prices (base CPM). Includes data engineering in PySpark, experiment tracking with WandB, cloud-integrated storage and querying with S3, AWS Glue, Amazon Athena and visual performance tracking with Grafana dashboards

## Architecture
![Untitled Diagram drawio](https://github.com/user-attachments/assets/450625ad-cf1e-42cc-9363-d87729ab280f)

## How to run?

Clone the repository

```bash
https://github.com/HarshVBhatt/ad-slot-reserve-price-prediction.git
```
## STEP 01- Create a conda environment after opening the repository

```bash
conda create -n reserve-price-prediction
```

```bash
conda activate reserve-price-prediction
```


## STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


## STEP 03- setup aws services

### 1. Login to AWS console.

### 2. Create IAM user for deployment and access key. Note these credentials 

	#Policy:

	1. AmazonAthenaFullAccess

	2. AmazonGrafanaAthenaAccess

	3. AmazonS3FullAccess

	4. AmazonGlueConsoleFullAccess


### 3. Set up a public s3 bucket with the name "reserve-price-prediction"

### 4. Configure to read/write to s3
```bash
pip install awscli
aws configure
```

- Enter the credentials you previously noted.

- Enter your default region for the region prompt

- Enter "json" for output prompt

Run the following to ensure s3 access
```bash
aws s3 ls
```

## STEP 04- Download the original data
- Download the original data from - https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction/data
- Convert the downloaded .zip files into a single file - raw_data.zip and upload to your github repository (License and competition rule compliance is your responsibility)

## OPTIONAL STEP
- Set the environment variable WANDB_API_KEY as your wandb api key


## STEP 05- Run the pipeline locally
```bash
python src/main.py
```

If the WANDB_API_KEY environment variable was not set earlier, you will be prompted for the key

You should see models and metadata being saved in the artifacts file

Once the pipeline completes, check s3 folder for successful file uploads

## STEP 06- Setup and run Glue crawler
- Create a new crawler with the bucket path being the one where the output parquets are stored
- Set up a new IAM role, if necessary, with S3FullAccess and GlueConsoleFullAccess
- Run the crawler
- A Data Catalog will be constructed on successful run of the crawler

## STEP 07 - Setup Athena to read and query the data catalog
- Open the query editor and make sure the Data Source dropdown has "AWSDataCatalog" selected
- The database field dropdown should have the respective database name
- Run a sample query to check successful connection

## STEP 08 - Setup Grafana
- Setup a new stack
- Install the AWS Athena plugin, if not previously installed
- Connect your AWS account to Grafana
- Add a new data source with an Athena connector
- If necessary, set up an IAM role to allow Grafana to connect to AWS
- Once the data source is added, you can build Dashboards using it


- Sample Grafana dashboard snapshot: https://harshvbhatt.grafana.net/dashboard/snapshot/uFEYCq35WUUJq8GRdXHqcZ2edvQCu178
