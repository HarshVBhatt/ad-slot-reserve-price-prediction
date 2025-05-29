from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

# def add_predictions(test_df, predictions):
#     pred_df = spark.createDataFrame(data = predictions, schema = ["Predictions"])
#     pred_df = pred_df.withColumn('id', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))

#     temp = test_df.select("*")
#     temp = temp.withColumn('id', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))

#     new_test_df = temp.join(pred_df, on=['id']).drop('id')
#     return new_test_df

# def clean_columns(original_df, new_df):
#     drop_columns = [c for c in new_df.columns if c not in original_df.columns]
#     drop_columns.remove("Prediction")
#     return new_df.drop(*drop_columns)

def inv_transform_target(orig_df, predictions):
    target_mean = orig_df.select(F.mean(F.col("CPM"))).collect()[0][0]
    target_std = orig_df.select(F.std(F.col("CPM"))).collect()[0][0]

    predictions_inv = predictions * target_std + target_mean
    return predictions_inv

def add_predictions(test_df, predictions):
    def fill_predictions(x):
        return predictions[x-1]
    fill_predictions_udf = F.udf(lambda x: predictions[x-1], IntegerType())

    temp = test_df.select("*")
    temp = temp.withColumn("num_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
    new_df = temp.withColumn("predictions", fill_predictions_udf("num_id"))

    return new_df

def get_original_columns(original_df, new_df):
    drop_columns = [c for c in new_df.columns if c not in original_df.columns]
    drop_columns.remove("Predictions")
    new_df = new_df.drop(*drop_columns)
    return new_df