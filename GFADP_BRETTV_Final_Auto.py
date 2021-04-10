# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:22:19 2019

@author: iloo
1) read in csv file
2) cluster
3) calculate attribute for each cluster
4) output csv file to be made into Klarf File

Programs:
    Final to be run automatically - this one.
    Sensitizing Clustering - edit eps and min
    Manual Classing for True Values - Automatic/Manual Classing
    Training for Algorithm - After Sensitizing Clustering + Manual Classing
    Test for Algorithm - Can be combined ^^
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pickle

def centr(gfa):
    cgfa = np.mean(gfa,axis=0)
    return cgfa

def gfalen(gfa):
    dist_arr = np.sqrt(np.sum(gfa**2,axis=1))
    coordA = gfa[np.argmax(dist_arr), :]
    gfa_moved = gfa[:] - coordA[:]
    dist_arr = np.sqrt(np.sum(gfa_moved**2,axis=1))
    coordB = gfa[np.argmax(dist_arr), :]
    gfalen = np.linalg.norm(coordA - coordB)
    return gfalen

def PolygonArea(xy):
    x = xy[:,0]
    y = xy[:,1]
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    area = 0.5*np.abs(main_area + correction)
    return area

def gfaarea(gfa):
    if all(gfa[:,0] == gfa[1,0]) or all(gfa[:,1] == gfa[1,1]):
        gfaarea = 0
    else:
        hull = ConvexHull(gfa)
        hull_pts = gfa[hull.vertices,:]
        gfaarea = PolygonArea(hull_pts)
    return gfaarea

def cluster(xy,epsn,mins):
    dbscan = DBSCAN(eps=epsn,min_samples=mins).fit(xy)
    gfaID = 1 + dbscan.labels_
    return gfaID 

def zone(gfaC):
    dist = np.sqrt(np.sum(np.square(gfaC)))
    return dist

def insertb4p(string):
    ind = string.index('.')
    string_n = string[:ind] + '1' + string[ind:]
    return string_n
    
def main(filename):
    df = pd.read_csv(filename,names=['x','y','class_pred'])
    xy = df.to_numpy()
    gfaID = cluster(xy,9,6) #EDIT EPS and MIN
    gfaID_noZ = np.unique(gfaID)[1:]
    if gfaID_noZ.size != 0:
        for i,gid in enumerate(gfaID_noZ):
            index = gfaID == gid
            gfa = xy[index]
            gfaC = centr(gfa)
            gfaleng = gfalen(gfa)
            if gfaleng > 1: #EDIT Value
                centroid = gfaC
                dist = zone(gfaC)
                leng = gfaleng
                area = gfaarea(gfa)
                num = len(gfa)
                attr_list = [centroid, dist, leng, area, num] #Make sure prediction model is in this order 
                pred_mdl = pickle.load(open('predict_model.pkl','rb')) #Predict!
                df.loc[index,'class_pred'] = pred_mdl.predict(attr_list) #1, 2, 3, cannot be nothing
    filename_n = insertb4p(filename)
    df.to_csv(filename_n,index=False) #Need to export as a Klarf though...
    return None

if __name__ == '__main__':
    filename = 'SOMEFILE.csv' #Probably take the most recent one. Keep running always
    main(filename)
