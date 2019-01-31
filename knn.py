import math
import operator

import pandas as pd
from sklearn.utils import shuffle


def get_euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        return 'Impossible! Should be one dimension!'
    n = len(v1)
    s = 0
    for i in range(n):
        if pd.isnull(v1[i]) or pd.isnull(v2[i]):
            continue
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
    max_v = d[0][1]
    max_item = d[0][0]
    for i in range(len(d)):
        if d[i][1] > max_v:
            max_v = d[i][1]
            max_item = d[i][0]
    return max_item


def knn(data, column, k=2, max_neihbours=10):
    feature_column = [col for col in data.columns if col != column]
    train_data = data[data[column].notnull()]
    test_data = data[data[column].isnull()]
    train_data_x = train_data[feature_column]
    train_data_y = train_data[column]
    test_data_x = test_data[feature_column]
    test_data_y = test_data[[column]]
    ss = test_data_x.index
    pred_values = list()
    j = 0
    for i in test_data_y.index:
        # train_data_i = shuffle(train_data_x).iloc[:max_neihbours]
        try:
            neighbours = getNeighbors(train_data_x.iloc[:max_neihbours], test_data_x.iloc[i], k)
        except:
            a = 5
        try:
            pred_value = predict(train_data_y.iloc[neighbours])
        except:
            s = [neighbours[i] in train_data_y.index for i in range(len(neighbours))]
            if sum(s) != k:
                pred_value = 0
        print(j)
        j += 1
        pred_values.append(pred_value)
        test_data_y.iloc[i] = pred_value
    r = pd.concat([test_data_x, test_data_y], axis=1)
    return pd.concat([train_data, r], axis=0)


filename = 'C:/Users/user/U_data_ML_homework_1/weatherAUS.csv'
df = pd.read_csv(filename)
df_ = knn(df, 'Pressure9am', k=5)
