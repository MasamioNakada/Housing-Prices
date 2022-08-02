import pandas as pd
import numpy as np

from wikiframe import Say, Extractor
from transform import one_hot_transform
from utils import txt_list

extractor = Extractor('data')

df_dict = extractor.extract_from_csv()

df_all = pd.concat([df_dict['house_train_raw'],df_dict['houses_test_raw']])

features = txt_list('features.txt')

df_all = df_all[features]

df_all = one_hot_transform(df_all,features)

train = df_all.iloc[:1460,:]
test = df_all.iloc[1460:,:]

if __name__ == "__main__":
    print(df_all.shape)  
    print(features)       
