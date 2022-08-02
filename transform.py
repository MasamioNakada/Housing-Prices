import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Numerical preprocesing
def normalize(df):
    '''
    Esta función normaliza los datos de un dataframe. Utiliza la función StandardScaler de sklearn.
    (z-score)

    Parameters 
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe (normalizado)
    '''
    col = df.keys()
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return pd.DataFrame(df, columns=col)

#Categorical Preprocessing

def encode_and_bind(df, feature):
    dummies = pd.get_dummies(df[feature])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature], axis=1)
    return(res) 

def one_hot_transform(df,features):
    '''
    Esta funcion tranforma las columnas categoricas de un dataframe a nuevas columnas numericas.

    Parameters
    ----------
    df : dataframe
    features : lista de las columnas a transformar

    Returns
    -------
    df : dataframe (transformado)
    '''
    for feature in features:
        if df.dtypes[feature] == np.dtype(object):
            df =  encode_and_bind(df,feature = feature)
    return df


