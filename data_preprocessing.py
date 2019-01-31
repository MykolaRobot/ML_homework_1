import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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


def delete_nan_rows(df, subset=None):
    df__ = df.copy()
    if subset == None:
        for index, raw in df__.iterrows():
            if sum(np.isnan(raw.values)) > 0:
                df__.drop(index, inplace=True)
            print(index)
    else:
        df_ = df__[subset]
        for index, raw in df_.iterrows():
            if sum(np.isnan(raw.values)) > 0:
                df__.drop(index, inplace=True)
    return df


def replace_null_values(df, col=None, replace_to='mean'):
    df_=df.copy()
    if replace_to == 'mean':
        repl = lambda x: np.nanmean(x)
    elif replace_to == 'median':
        repl = lambda x: np.nanmedian(x)
    elif replace_to == 'mode':
        repl = ''
    else:
        return 'parameter replace_to has taken unrecognised value'

    def f(raw, df, c):
        if c is None:
            columns = df.columns
        else:
            columns = c
        for column in columns:
            if pd.isnull(raw[column]):
                raw[column] = repl(list(df[column].values))
        return raw

    return df.apply(f, args=(df_, col,), axis=1)




def standart_scaler(df, col=None):
    if col is None:
        columns = df.columns
    else:
        columns = col
    for column in columns:
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


def replace_null_by_linear_regression(df, target_column, col_based_on=None):
    # check if column contains missing values
    columns = df.columns  # use linear regression
    # rows that we will predict
    df_train = df[df[target_column].notnull()]
    df_pred = df[df[target_column].isnull()]
    if col_based_on is None:
        X_col = [col for col in columns if col != target_column]
    else:
        X_col = col_based_on
    X_train = df_train[X_col]
    y_train = df_train[target_column]
    lin_model = LinearRegression().fit(X_train, y_train)
    X_pred = df_pred[X_col]
    pred_values = lin_model.predict(X_pred)
    df_pred[target_column] = pred_values
    df1 = pd.concat([df_pred, df_train], axis=0)
    return df1


def my_drop(df, axis=1):
    if axis == 1:
        index = pd.isnull(df).any(1).nonzero()[0]
        df = df.drop(index)
        return df
    elif axis == 0:
        for i in df:
            if np.isnan(df[i]).any():
                df = df.drop(i, axis=1)
        return df
    else:
        return -1


filename = 'C:/Users/user/U_data_ML_homework_1/weatherAUS.csv'
df = pd.read_csv(filename).iloc[10000:20000]
# df = replace_null_values(df,col=['MinTemp', 'MaxTemp'], replace_to='mean')
# print(df.info())
# # df = replace_null_by_linear_regression(df, 'Rainfall', col_based_on=['MinTemp', 'MaxTemp'])
# # print(df.info())
# # print(df.head(10))
