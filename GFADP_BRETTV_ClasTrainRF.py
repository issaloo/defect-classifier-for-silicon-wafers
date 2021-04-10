# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:13:19 2019

@author: iloo
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

def testsplit(dataset,rand_stat):
    X_cln = dataset.drop(dataset.columns[-1],axis=1)
    y_cln = dataset[dataset.columns[-1]]
    X_train,X_test,y_train,y_test = train_test_split(X_cln,y_cln,test_size=0.2,
                                                     random_state=rand_stat)        
    return X_train, X_test, y_train, y_test

def main(dataset):
    X_train, X_test, y_train, y_test = testsplit(dataset,15)
    max_depth = [int(x) for x in np.linspace(2,12,num=1)]
    max_depth.append(None)
    random_grid = {'n_estimators':[int(x) for x in np.linspace(start=10, stop=100,num=5)],
                   'max_features': ['auto','sqrt'],
                   'max_depth': max_depth,
                   'min_samples_split': [2,5,10],
                   'min_samples_leaf': [1,4,8]}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                   n_iter = 200, cv = 3, verbose=2, random_state=42,
                                   n_jobs = -1)
    rf_random.fit(X_train,y_train)
    y_pred = rf_random.predict(X_test)
    print('Test Confusion Matrix')
    print(metrics.confusion_matrix(y_test,y_pred))
    target_names = ['Streak','Focus Spot']
    print('Test Classification Report')
    print(metrics.classification_report(y_test,y_pred,target_names=target_names))

    pickle.dump(rf_random, open('rf_Predict_model.pkl','wb'))

if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    trainingset = pd.read_csv('BRETTV_TD.csv')
    main(trainingset)