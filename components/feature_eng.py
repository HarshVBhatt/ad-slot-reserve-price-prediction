import pyspark.sql.functions as F
from pyspark.sql.functions import col
from itertools import chain
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline

# Non-zero division function
def divide(n, d):
    if d:
        return n / d 
    return 0
divide_udf = F.udf(divide)

# Function to construct frequency dictionary
def get_freq_dict(df, c):
    aggregated_df = df.groupBy(F.col(c)).agg(F.count("*").alias("count")).collect()
    col_list = [row.asDict() for row in aggregated_df]

    freq_dict = {}
    for row in col_list:
        freq_dict[row[c]] = row["count"]

    return freq_dict

# Function to create mapper
def get_mapper(freq_dict):
    return F.create_map([F.lit(x) for x in chain(*freq_dict.items())])

def feature_engineering(df):
    # Get column types
    unq_counts = {}
    for c in df.columns:
        count = df.select(c).distinct().count()
        unq_counts[c] = count

    drop_cols = [c for c,v in unq_counts.items() if v == 1]
    cat_cols = [c for c,v in unq_counts.items() if "_id" in c and c not in drop_cols]
    high_card_cat_cols = [c for c,v in unq_counts.items() if c in cat_cols and v > 15]
    low_card_cat_cols = [c for c in cat_cols if c not in high_card_cat_cols]
    num_cols = [c for c,v in unq_counts.items() if c not in drop_cols and c not in cat_cols and c != "date"]

    print(f"""Columns to drop: {drop_cols}
    Low Cardinatlity Categorical columns: {low_card_cat_cols}
    High Cardinatlity Categorical columns: {high_card_cat_cols}
    Numerical columns: {num_cols}""")

    # Drop columns
    print("Dropping columns")
    for c in drop_cols:
        df = df.drop(c)

    # Engineer CPM column, drop outliers
    print("Constructing CPM column")
    df = df.withColumn("CPM", divide_udf(df["total_revenue"] * 100, df["measurable_impressions"]) * 1000)

    quantile_threshold = df.approxQuantile("CPM", [0.95], 0.01)[0]
    df = df.filter(df["CPM"] <= quantile_threshold)

    # Engineer Viewable Impressions to Measurable Impressions ratio
    print("Constructing v/m ratio")
    df = df.withColumn("v/m ratio", divide_udf(df["viewable_impressions"], df["measurable_impressions"]))

    # Drop correlated columns
    print("Dropping correlated columns")
    num_cols = [c for c in num_cols if c not in ["total_revenue", "total_impressions", "viewable_impressions"]]

    # Frequency encode columns - geo_id, ad_unit_id, order_id
    print("Frequency encoding columns")
    geo_dict = get_freq_dict(df, "geo_id")
    geo_mapper = get_mapper(geo_dict)

    adunit_dict = get_freq_dict(df, "ad_unit_id")
    adunit_mapper = get_mapper(adunit_dict)

    order_dict = get_freq_dict(df, "order_id")
    order_mapper = get_mapper(order_dict)

    df = (df
      .withColumn("geo_id", geo_mapper[F.col("geo_id")])
      .withColumn("ad_unit_id", adunit_mapper[F.col("ad_unit_id")])
      .withColumn("order_id", order_mapper[F.col("order_id")])
      )
    
    # Prepare advertiser_id column for one-hot encoding
    print("Preparing column for One Hot Encoding")
    adv_id_grouped = df.groupBy(F.col("advertiser_id")).agg(F.count("*").alias("count"))
    total_vals = adv_id_grouped.select(F.sum("count")).collect()[0][0]
    adv_id_grouped = adv_id_grouped.withColumn("pct", F.round(adv_id_grouped["count"] * 100/ total_vals, 2))
    adv_id_grouped = adv_id_grouped.filter(F.col("pct") > 1)
    freq_advertisers = adv_id_grouped.select(F.col("advertiser_id")).rdd.flatMap(lambda x:x).collect()

    def get_advertiser(x):
        if x in freq_advertisers:
            return x
        return 0
    get_advertiser_udf = F.udf(get_advertiser)

    df = df.withColumn("advertiser_id", get_advertiser_udf(F.col("advertiser_id")))

    # One-hot encode
    print("One-hot encoding columns")
    oh_columns = low_card_cat_cols + ["advertiser_id"]

    indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in oh_columns]
    encoder = OneHotEncoder(inputCols=[col+"_idx" for col in oh_columns], outputCols=[col+"_ohe" for col in oh_columns])
    oh_pipeline = Pipeline(stages = indexers + [encoder])
    df_encoded = oh_pipeline.fit(df).transform(df)

    # MinMax scale frequency encoded features
    print("MinMax Scaling columns")
    minmax_cols = [c for c in high_card_cat_cols if c not in oh_columns]
    minmax_assembler = VectorAssembler(inputCols=minmax_cols, outputCol = "minmax_features_vec")
    minmax_scaler = MinMaxScaler(inputCol="minmax_features_vec", outputCol="minmax_features")
    minmax_pipeline = Pipeline(stages=[minmax_assembler, minmax_scaler])
    df_scaled = minmax_pipeline.fit(df_encoded).transform(df_encoded)

    # Standard scale numerical features
    print("Standard Scaling columns")
    stdscalar_assembler = VectorAssembler(inputCols=num_cols, outputCol="stdscaler_features_vec")
    std_scaler = StandardScaler(inputCol="stdscaler_features_vec", outputCol="stdscaler_features")
    stdscaler_pipeline = Pipeline(stages=[stdscalar_assembler,std_scaler])
    df_scaled = stdscaler_pipeline.fit(df_scaled).transform(df_scaled)

    # Scale target variable - CPM
    print("Scaling target feature")
    target_scalar_assembler = VectorAssembler(inputCols=["CPM"], outputCol="target_features_vec")
    target_scaler = StandardScaler(inputCol="target_features_vec", outputCol="target_scaler_features")
    target_scalar_pipeline = Pipeline(stages = [target_scalar_assembler, target_scaler])
    df_transformed = target_scalar_pipeline.fit(df_scaled).transform(df_scaled)

    # Assemble features
    print("Assembling features")
    all_feature_cols = [col+"_ohe" for col in oh_columns] + ["minmax_features", "stdscaler_features"]
    assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")
    df_final = assembler.transform(df_transformed)
    df_final = df_final.sort(F.col("date").asc())

    print("All processes completed")
    return df_final

    