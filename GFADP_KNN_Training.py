# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:39:46 2019

@author: iloo
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

def testsplit(dataset,rand_stat):
    comp = (dataset['GFA_ID'] != 0) & (dataset['GFA_Class'] !=3)
    scaler = MinMaxScaler()
    X_cln = scaler.fit_transform(dataset.loc[comp,'GFA_AREA':'GFA_NUM'])
    y_cln = dataset.loc[comp,'GFA_Class']
    
    X_train,X_test,y_train,y_test = train_test_split(X_cln,y_cln,test_size=0.2,
                                                     random_state=rand_stat)        
    return X_train, X_test, y_train, y_test

def kNN_Training(trainingset):
    X_train, X_test, y_train, y_test = testsplit(trainingset,15)
    
    k_range = range(2,20)
    scores_list = []
    for k in k_range:
        kNN_model = KNeighborsClassifier(n_neighbors=k)
        kNN_model.fit(X_train,y_train)
        y_pred = kNN_model.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
    plt.figure()
    plt.plot(k_range,scores_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()
    
    k_Opt = k_range[np.argmax(scores_list)]
    return k_Opt

def kNN_Pickling(trainingset,k_Opt):
    X_train, X_test, y_train, y_test = testsplit(trainingset,5)
    kNN_Opt = KNeighborsClassifier(n_neighbors = k_Opt)
    kNN_Opt.fit(X_train,y_train)
    y_pred = kNN_Opt.predict(X_test)
    
    pickle.dump(kNN_Opt, open('KNN_Predict_model.pkl','wb'))
    print('--- Optimal K = %d ---' %(k_Opt))
    print('K-Test Classification Report')
    target_names = ['Streak','Focus Spot']
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred,target_names=target_names))
    
def predictGFA(testset):
    comp = (testset['GFA_ID'] !=0) & (testset['GFA_Class'] !=3)
    print(comp)
    scaler = MinMaxScaler()
    X_cln = scaler.fit_transform(testset.loc[comp,'GFA_AREA':'GFA_NUM'])
    print(X_cln)
    y_cln = testset.loc[comp,'GFA_Class']
    KNN_Predict_model = pickle.load(open('KNN_Predict_model.pkl','rb'))
    y_pred = KNN_Predict_model.predict(X_cln)
    print('Validation Classification Report')    
    target_names = ['Streak','Focus Spot']
    print(metrics.confusion_matrix(y_cln,y_pred))
    print(metrics.classification_report(y_cln,y_pred,target_names=target_names))
    
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    trainingset = pd.read_csv('TrainingData.csv')
    k_Opt = kNN_Training(trainingset)
    kNN_Pickling(trainingset,k_Opt)
    testset = pd.read_csv('TestData.csv')
    predictGFA(testset)