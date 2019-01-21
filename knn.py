import pandas as pd
import math
import operator
import numpy as np

d = {'col1': [np.nan, 2, 3, 5, 7, 8], 'col2': [0, 3, 2, 3, 4, 5], 'col3': [1, 2, 3, 4, 5, 6]}
data = pd.DataFrame(data=d)


def get_euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        return 'Impossible! Should be one dimension!'
    n = len(v1)
    s = 0
    for i in range(n):
        s += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(s)


def getNeighbors(train_data, test_example, k=2):
    distances = list()
    for index, raw in train_data.iterrows():
        distances.append((index, get_euclidean_distance(raw, test_example)))
    distances.sort(key=operator.itemgetter(1))
    return [distances[i][0] for i in range(k)]


def predict(k_data):
    k_data = list(k_data.values)
    # find the most frequent value
    d = [(item, k_data.count(item)) for item in k_data]
    d.sort(key=operator.getitem(1))
    return d[0]


def knn(data, column, k=2):
    feature_column = [col for col in data.columns if col != column]
    train_data = data[data[column].notnull()]
    test_data = data[data[column].isnull()]
    train_data_x = train_data[feature_column]
    train_data_y = train_data[column]
    test_data_x = test_data[feature_column]
    pred_values = list()
    for i in test_data_x.index:
        neighbours = getNeighbors(train_data_x, test_data_x.iloc[i], k)
        pred_value = predict(train_data_y.iloc[neighbours])
        pred_values.append(pred_value)
        test_data.iloc[i][column] = predict(pred_values)
    return pd.concat([train_data, test_data], axis=0)


dd = knn(data, 'col1', 2)
predict(dd)
