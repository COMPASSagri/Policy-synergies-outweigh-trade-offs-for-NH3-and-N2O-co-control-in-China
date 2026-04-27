# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:21:37 2026

@author: sera
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)   
pd.set_option('display.float_format', lambda x: '%.5f' % x) 
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
import seaborn as sns  
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import scipy.stats
from time import *
from tqdm import tnrange, tqdm_notebook
from time import sleep
from matplotlib.mathtext import _mathtext as mathtext
from sklearn.utils import resample
from sklearn.ensemble import VotingRegressor
mathtext.FontConstantsBase.sup1 = 0.3 

#%%读取文件
nh3 = pd.read_excel('/data/nh3XY.xlsx')
nh3df=nh3.copy()
#%%乱序
nh3df = sklearn.utils.shuffle(nh3df,random_state=2022) 
nh3df.index = range(0,len(nh3df))
#%%X和Y
X = nh3df[['STP', 'Prec', 'Tmp', 'soc', 'tn', 'pH', 'bd', 'clay', 'cec', 'Nrate', 'UOA', 'ABC', 'Others', 'Manure', 'Compound', 'SBC', 'DPM', 'Rice', 'Wheat', 'Maize', 'Other_upland', 'Vegetable']]
Y = nh3df['EF']
#%%寻找最佳模型
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) 

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor  # Make sure you have XGBoost installed

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regression': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'XGBoost Regressor': XGBRegressor()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

# Display results
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")
    
#%%寻找最佳参数组合
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

#参数可调整
param_grid = {
    'n_estimators': [100,200,500],  # List of integers to try
    'max_features': [None,'sqrt'],  # 最大特征数
    'min_samples_split': [2],
    'min_samples_leaf': [3],
    'max_leaf_nodes':[None],
    'max_depth': [25]
}

# Initialize Random Forest Regressor
rf = RandomForestRegressor(random_state=42,oob_score=True,bootstrap=True)

# Set up GridSearchCV
grid = GridSearchCV(estimator=rf, 
                    param_grid=param_grid, 
                    cv=10, 
                    n_jobs=-1, 
                    return_train_score=False)

# Fit the grid search to your data (e.g., X, y)
grid.fit(x_train, y_train)

 # Get best parameters
best_params = grid.best_params_

#构建模型
rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                 max_features=best_params['max_features'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 max_leaf_nodes=best_params['max_leaf_nodes'],
                                 max_depth=best_params['max_depth'],random_state=42,oob_score=True,bootstrap=True)

#数据拟合
rf_model.fit(x_train, y_train)

# Predict on training and test sets
y_train_pred = rf_model.predict(x_train)
y_test_pred = rf_model.predict(x_test)

# Calculate R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate MSE and RMSE
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

# Calculate MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Print results
print(f"Training R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Training RMSE: {rmse_train:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")
print(f"Training MAE: {mae_train:.3f}")
print(f"Test MAE: {mae_test:.3f}")

# Get the best parameters and the best score
best_params = grid.best_params_
best_score = grid.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


#%%用所有数据再训练一个模型
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import cross_val_score

rf_model.fit(X, Y)
scores = cross_val_score(rf_model, X, Y, cv=10)
scores.mean()


