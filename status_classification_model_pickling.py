import pickle

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

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


dfc = df.copy()

dfc = df[df['status'].isin(['Won', 'Lost'])]

### Data Transformation
## Label Encoding
#use Label encoder to convert categorical data into numerical data.
le = LabelEncoder()
dfc.status = le.fit_transform(dfc[['status']])
dfc['item type'] = le.fit_transform(dfc[['item type']])



#split data into X, y
X = dfc[['quantity tons','selling_price','item type','application','thickness','width','country','customer','product_ref']]
y = dfc['status']

## Modelling
#split data into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)


## Scaling
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS.fit_transform(X_train)

## Classification

#xgb_model = xgb.XGBClassifier(n_estimators=100,  learning_rate=0.5, max_depth= 10,objective="binary:logistic", random_state=42)
#xgb_model.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors =5)
knn.fit(X_train, y_train)
        
        
pickle.dump(knn, open('status_classification_model.pkl','wb'))