# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:46:02 2019
@author: iloo
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pickle

def testsplit(dataset,rand_stat):
    comp = (dataset['GFA_ID'] != 0) & (dataset['GFA_Class'] !=3)
    X_cln = dataset.loc[comp,'GFA_AREA':'GFA_NUM']
    y_cln = dataset.loc[comp,'GFA_Class']
    
    X_train,X_test,y_train,y_test = train_test_split(X_cln,y_cln,test_size=0.25
                                                     , random_state=rand_stat)        
    return X_train, X_test, y_train, y_test

def rf_Training(dataset):
    X_train, X_test, y_train, y_test = testsplit(dataset,18)
    #Number of trees in random Forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100,num=5)]
    #Number of features to consider at every split
    max_features = ['auto','sqrt']
    #Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2,12,num=1)]
    max_depth.append(None)
    #Minimum number of samples required to split a node
    min_samples_split = [2,5,10]
    #Minimum number of samples required at each leaf node
    min_samples_leaf = [4,8]
    #Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    random_grid = {'n_estimators':n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                   n_iter = 200, cv = 3, verbose=2, random_state=42,
                                   n_jobs = -1)
    rf_random.fit(X_train,y_train)
    y_pred = rf_random.predict(X_test)
    print('Test Classification Report')
    target_names = ['Streak','Focus Spot']
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred,target_names=target_names))
    pickle.dump(rf_random, open('rf_Predict_model.pkl','wb'))

def predictGFA(testset):
    comp = (testset['GFA_ID'] != 0) & (testset['GFA_Class'] !=3) & (testset['GFA_Class'] !=0)
    X_cln = testset.loc[comp,'GFA_AREA':'GFA_NUM']
    y_cln = testset.loc[comp,'GFA_Class']
    rf_Predict_model = pickle.load(open('rf_Predict_model.pkl','rb'))
    y_pred = rf_Predict_model.predict(X_cln)
    print('Validation Classification Report')
    target_names = ['Streak','Focus Spot']
    print(metrics.confusion_matrix(y_cln,y_pred))
    print(metrics.classification_report(y_cln,y_pred,target_names=target_names))

if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    trainingset = pd.read_csv('TrainingData.csv')
    rf_Training(trainingset)    
    testset = pd.read_csv('TestData.csv')
    predictGFA(testset)