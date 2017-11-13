import pickle
import pandas as pd
import numpy as np
import random


def get_data_sequence(datam, columns, n_prev=10):

    docX, docY = [], []
    lendata = len(datam)
    for i in range(lendata - n_prev):
        docX.append(datam[columns].iloc[i:i + n_prev].as_matrix())
        docY.append(datam[['Sales']].iloc[i + n_prev])
    return np.array(docX), np.array(docY)

def train_test_split(df, test_size=0.1):
    n_examples = int(round(len(df) * (1 - test_size)))
    data_train = df.iloc[0:n_examples]
    data_test = df.iloc[n_examples:]
    return (data_train, data_test)


def store_results(dataframe, output_file):
    dataframe[['Id', 'Sales']].to_csv(output_file, index=False)
