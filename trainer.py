# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

#from keras.callbacks import ModelCheckpoint
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten

#from xgboost import XGBRegressor

#this code separates all_matches.csv into test and train files
df_processed = pd.read_csv('all_matches_preprocessed.csv')
df_clean = df_processed.drop(['match_id', 'start_date'], axis = 1)
# Labels are the values we want to predict
labels = np.array(df_clean['total_run'])
# Remove the labels from the features
# axis 1 refers to the columns
df_clean = df_clean.drop('total_run', axis = 1)
# Saving feature names for later use
feature_list = list(df_clean.columns)
# Convert to numpy array
df_clean = np.array(df_clean)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_clean, labels, test_size = 0.1, random_state = 42)
#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)


def train_rf():
    n_trees = np.arange(10,101,1)
    err = []
    for item in n_trees:
        rf = RandomForestRegressor(n_estimators = item, random_state = 42)
        rf.fit(train_features, train_labels);
        pred_rf = rf.predict(test_features)
        err_val = np.subtract(pred_rf, test_labels)
        squared = np.square(err_val)
        ms = squared.sum()
        err.append(ms)
    err_arr = np.array(err)
    DT = n_trees[err.index(np.min(err_arr))]
    #plt.plot(n_trees,err_arr,'b-')
    # Instantiate model with decision trees
    rf = RandomForestRegressor(n_estimators = DT, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    joblib.dump(rf, 'rf_m.pkl')
def train_svr():

    svr_rbf = SVR(kernel='rbf')
    svr_lin = SVR(kernel='linear')
    svr_poly = SVR(kernel='poly')
    svr_rbf_m = svr_rbf.fit(train_features, train_labels)
    svr_lin_m = svr_lin.fit(train_features, train_labels)
    svr_poly_m = svr_poly.fit(train_features, train_labels)
    
    joblib.dump(svr_rbf_m, 'svr_rbf_m.pkl')
    joblib.dump(svr_lin_m, 'svr_lin_m.pkl')
    joblib.dump(svr_poly_m, 'svr_poly_m.pkl')

def train_models():
    train_rf()
    train_svr()

train_models()
