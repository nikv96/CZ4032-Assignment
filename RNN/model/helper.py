import numpy as np


def obtain_data_sequence(new_data, columns, num_of_prevs=10):

    Xdoc, Ydoc = [], []
    lendata = len(new_data)
    for i in range(lendata - num_of_prevs):
        Xdoc.append(new_data[columns].iloc[i:i + num_of_prevs].as_matrix())
        Ydoc.append(new_data[['Sales']].iloc[i + num_of_prevs])
    return np.array(Xdoc), np.array(Ydoc)


def split_test_train(df, test_size=0.1):
    n_patterns = int(round(len(df) * (1 - test_size)))
    train_data = df.iloc[0:n_patterns]
    test_data = df.iloc[n_patterns:]
    return (train_data, test_data)


def store_results(dataframe, out_file):
    dataframe[['Id', 'Sales']].to_csv(out_file, index=False)
