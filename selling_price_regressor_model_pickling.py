#import necessary libraries

import pickle

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#import regression algorithm.
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder



#Load the dataset
df = pd.read_csv('Copper_Set.csv')



#Data cleaning or preparation

df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

df['material_ref'].fillna('unknown', inplace=True)
df = df.dropna()

dfr=df.copy()

a = dfr['selling_price'] <= 0
dfr.loc[a, 'selling_price'] = np.nan

a = dfr['quantity tons'] <= 0
dfr.loc[a, 'quantity tons'] = np.nan

dfr['selling_price_log'] = np.log(dfr['selling_price'])
dfr['quantity tons_log'] = np.log(dfr['quantity tons'])
dfr['thickness_log'] = np.log(dfr['thickness'])


### Data Transformation
## Label Encoding
le = LabelEncoder()
dfr.status = le.fit_transform(dfr[['status']])
dfr['item type'] = le.fit_transform(dfr[['item type']])

dfr1 = dfr.dropna()
df_data = dfr1.copy()


X=df_data[['quantity tons_log','status','item type','application','thickness_log','width','country','customer','product_ref']]
y=df_data['selling_price_log']


## Modelling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

## Scaling
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS.fit_transform(X_train)

## Regression

dtr = DecisionTreeRegressor()
# hyperparameters
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_



pickle.dump(best_model, open('selling_price_regresser_model.pkl','wb'))