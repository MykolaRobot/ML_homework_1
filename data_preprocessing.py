import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

d = {'col1': [np.nan, 2, 3, 5, 7, 8], 'col2': [0, 3, 2, 3, 4, 5], 'col3': [1, 2, 3, 4, 5, 6]}
data = pd.DataFrame(data=d)


def delete_empty_axis(df, axis=0):
    if not isinstance(df, pd.DataFrame):
        return 'Need DataFrame'
    if axis == 0:
        # by rows
        for index, raw in df.iterrows():
            values = list(raw.values)
            if sum(np.isnan(values)) == len(values):
                # remove raw
                df.drop(index, inplace=True)

    elif axis == 1:
        columns = df.columns
        for column in columns:
            values = list(df[column].values)
            if sum(np.isnan(values)) == len(values):
                # remove column
                df.drop(column, axis=1, inplace=True)
    else:
        return 'Incorrect value of axis parameter'
    return df


def replace_null_values(df, replace_to='mean'):
    if replace_to == 'mean':
        repl = lambda x: np.nanmean(x)
    elif replace_to == 'median':
        repl = lambda x: np.nanmedian(x)
    elif replace_to == 'mode':
        repl = ''
    else:
        return 'parameter replace_to has taken unrecognised value'
    columns = df.columns
    for i in df.index:
        raw = df.iloc[i]
        for column in columns:
            if pd.isnull(raw[column]):
                a = repl(list(df[column].values))
                raw[column] = a
                df.iloc[i] = raw
    return df


def standart_scaler(df):
    for column in df.columns:
        mean = np.mean(df[column].values)
        std = np.std(df[column].values)
        df[column] = (df[column] - mean) / std
    return df


def min_max_scaler(df):
    for column in df.columns:
        x_min = min(df[column].values)
        x_max = max(df[column].values)
        df[column] = (df[column] - x_min) / (x_max - x_min)
    return df


def replace_null_by_linear_regression(df, target_column):
    # check if column contains missing values
    columns = df.columns  # use linear regression
    # rows that we will predict
    df_train = df[df[target_column].notnull()]
    df_pred = df[df[target_column].isnull()]
    X_col = [col for col in columns if col != target_column]
    X_train = df_train[X_col]
    y_train = df_train[target_column]
    lin_model = LinearRegression().fit(X_train, y_train)
    X_pred = df_pred[X_col]
    pred_values = lin_model.predict(X_pred)
    df_pred[target_column] = pred_values
    df1 = pd.concat([df_pred, df_train], axis=0)
    return df1


