# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:56:19 2019

@author: iloo
1) Read in all .csv file from folder
2) Sensitize clustering
3) Manually choose
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cluster(xy,epsn,mins):
    dbscan = DBSCAN(eps=epsn,min_samples=mins).fit(xy)
    gfaID = 1 + dbscan.labels_
    return gfaID

def main(filename,epsrange,minrange):
    #get one wafer map, iterate through eps to see the differences
    #iterate through minimum to see the differences
    df = pd.read_csv(filename,names=['x','y'])
    xy = df.to_numpy()
    for epsv in epsrange:
        for minv in minrange:
            gfaID = cluster(xy,epsv,minv)
            plt.scatter(xy[:,0],xy[:,1],s=5,c=gfaID)
            plt.ylim((-150,150))
            plt.xlim((-150,150))
            plt.draw()
            plt.title('EPS:{},MIN:{}'.format(epsv,minv))
    return None

if __name__ == '__main__':
    filename = 'SOME.csv'
    epsrange = list(range(3,12))
    minrange = list(range(100,300,3))
    main(filename,epsrange,minrange)