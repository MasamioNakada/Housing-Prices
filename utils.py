import numpy as np 
def txt_list(path):
    with open(path, "r") as f:
        features = " ".join([l.rstrip("\n") for l in f]) 
        return features.split()

def get_int_col(df):
    int_col = []
    for col in df.keys():
        if df.dtypes[col] == np.dtype(np.int64):
            int_col.append(col)
    return int_col

def get_x(df):
    int_col = []
    for col in df.keys():
        if col != 'SalePrice':
            int_col.append(col)
    return int_col

