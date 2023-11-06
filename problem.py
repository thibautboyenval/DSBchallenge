import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_train_data(path="."):
    f_name = "train.parquet"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.parquet"
    return _read_data(path, f_name)

def _encode_data(X):
    X = X.copy()
    X.drop(columns = ['counter_name', 'site_name', 'site_id',
       'counter_installation_date', 'counter_technical_id',
       ], inplace=True)

    X.loc[:, "week"] = X["date"].dt.isocalendar().week
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    if "coordinates" in X.columns:
        X.drop(columns=["coordinates"], inplace=True)
    le = LabelEncoder()
    X['counter_id'] = le.fit_transform(X['counter_id'])
    
    return X.drop(columns=["date"])

def _encode_data2(X):
    X = X.copy()
    X.drop(columns = ['counter_name', 'site_name', 'site_id',
       'counter_installation_date', 'counter_technical_id',
       ], inplace=True)

    X.loc[:, "week"] = X["date"].dt.isocalendar().week
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    if "coordinates" in X.columns:
        X.drop(columns=["coordinates"], inplace=True)
    le = LabelEncoder()
    X['counter_id'] = le.fit_transform(X['counter_id'])

    X['week_sin'] = np.sin(2 * np.pi * X['week'] / 53)
    X['week_cos'] = np.cos(2 * np.pi * X['week'] / 53)

    # Normalize the values to the range [-1, 1]
    X['week_sin'] = (X['week_sin'] - X['week_sin'].min()) / (X['week_sin'].max() - X['week_sin'].min()) * 2 - 1
    X['week_cos'] = (X['week_cos'] - X['week_cos'].min()) / (X['week_cos'].max() - X['week_cos'].min()) * 2 - 1

    
    return X.drop(columns=["date", "week"])


def get_transformed_data(path="."):
    X_train, y_train = get_train_data(path)
    X_test, y_test = get_test_data(path)
    X_train = _encode_data(X_train)
    X_test = _encode_data(X_test)
    return X_train, y_train, X_test, y_test

