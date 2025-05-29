import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pyspark.sql.functions as F

def split_train_test(data, date = "2019-06-23"):
    train_df = data.filter(F.col("date") < date)
    test_df = data.filter(F.col("date") >= date)
    return train_df, test_df

def split_X_y(data):
    X = np.array(data.select("features").rdd.map(lambda x: x[0].toArray()).collect())
    y = np.array(data.select(["target_scaler_features"]).rdd.map(lambda x: x[0]).collect())
    return X,y

def train_model(X_train, y_train, **kwargs):
    parameters = {}
    for key, val in kwargs.items():
        parameters[key] = val

    model = xgb.XGBRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model, parameters

def get_predictions(model, X_test):
    return model.predict(X_test)

def score(y_test, y_pred):
    return root_mean_squared_error(y_test, y_pred)