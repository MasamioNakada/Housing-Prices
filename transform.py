import pandas as pd
import numpy as np

def encode_and_bind(df, feature):
    dummies = pd.get_dummies(df[feature])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature], axis=1)
    return(res) 

def one_hot_transform(df,features):
    for feature in features:
        if df.dtypes[feature] == np.dtype(object):
            df =  encode_and_bind(df,feature = feature)
    return df


