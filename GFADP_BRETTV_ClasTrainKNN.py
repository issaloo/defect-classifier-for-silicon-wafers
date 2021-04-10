# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:49:14 2019

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
    scaler = MinMaxScaler()
    X_cln = scaler.fit_transform(dataset.drop(dataset.columns[-1],axis=1))
    y_cln = dataset[dataset.columns[-1]]
    X_train,X_test,y_train,y_test = train_test_split(X_cln,y_cln,test_size=0.2,
                                                     random_state=rand_stat)        
    return X_train, X_test, y_train, y_test

def main(trainingset,k_range):
    X_train, X_test, y_train, y_test = testsplit(trainingset,15)
    f1_score, recall_score, precision_score = [], [], []
    for k in k_range:
        kNN_model = KNeighborsClassifier(n_neighbors=k)
        kNN_model.fit(X_train,y_train)
        y_pred = kNN_model.predict(X_test)
        f1_score.append(metrics.f1_score(y_test,y_pred))
        recall_score.append(metrics.recall_score(y_test,y_pred))
        precision_score.append(metrics.precision_score(y_test,y_pred))
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,size=(7,7),sharex=True)
    fig.suptitle('Prediction Scores')
    ax1.plot(k_range,f1_score)
    ax1.set_title('F1')
    ax2.plot(k_range,recall_score)
    ax2.set_title('Recall')
    ax3.plot(k_range,precision_score)
    ax3.set_title('Precision')
    k_Opt = k_range[np.argmax(f1_score)]
    print('--- Optimal K = {} ---'.format(k_Opt))
    kNN_Opt = KNeighborsClassifier(n_neighbors = k_Opt)
    kNN_Opt.fit(X_train,y_train)
    y_pred = kNN_Opt.predict(X_test)
    
    print('Test Confusion Matrix')
    target_names = ['Streak','Focus Spot'] #Edit Target names
    print(metrics.confusion_matrix(y_test,y_pred))
    print('Test Classification Report')
    print(metrics.classification_report(y_test,y_pred,target_names=target_names))

    pickle.dump(kNN_Opt, open('KNN_Predict_model.pkl','wb'))
    
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    trainingset = pd.read_csv('BRETTV_TD.csv') #Edit CSV
    k_range = range(2,20)
    main(trainingset,k_range)