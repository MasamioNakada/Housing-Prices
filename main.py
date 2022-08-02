import pandas as pd
import numpy as np

from wikiframe import Say, Extractor
from transform import normalize, one_hot_transform
from utils import txt_list, get_x

warnings.simplefilter("ignore")

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.linear_model import HuberRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score




extractor = Extractor('data')

df_dict = extractor.extract_from_csv()

df_all = pd.concat([df_dict['house_train_raw'],df_dict['houses_test_raw']])

features = txt_list('features.txt')

df_all = df_all[features]

df_all = one_hot_transform(df_all,features)
df_all = normalize(df_all)

train = df_all.iloc[:1460,:]
test = df_all.iloc[1460:,:]

X = train[get_x(df_all)]
y = train['SalePrice']

estimators = {
    'ElasticNet': ElasticNet(random_state=0),
    'Lasso': Lasso(alpha=0.2),
    'Ridge': Ridge(alpha=1),
    'Huber': HuberRegressor(epsilon=1.46,fit_intercept=True),
    'RandomForest': RandomForestRegressor(),
    'DecisionTree': DecisionTreeRegressor()
}

if __name__ == "__main__":
    for name, estimator in estimators.items():
        score = cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_log_error')
        print(name, np.abs(score).min())  

    random_params = {'n_estimators': [300,400,500,600],  
               'max_features': ['sqrt'],  
               'max_depth': [ 60, 70, 80, 90, 100,], 
               'min_samples_split':  [2, 5, 10], 
               'min_samples_leaf': [1, 2, 4], 
               'bootstrap': [True, False]}

    score_rand = RandomizedSearchCV(RandomForestRegressor(), random_params, cv=3, scoring='neg_mean_squared_log_error',n_iter=20).fit(X, y)  
    score = cross_val_score(RandomForestRegressor(n_estimators=400,max_features='sqrt',max_depth=70,min_samples_split=5,min_samples_leaf=2,bootstrap=True), X, y, cv=5, scoring='neg_mean_squared_log_error').fit(X,y)