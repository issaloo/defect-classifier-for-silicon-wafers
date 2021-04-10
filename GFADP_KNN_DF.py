# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:46:01 2019

@author: iloo
"""
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
from scipy.spatial import ConvexHull

def centr(gfa):
    cgfa = np.mean(gfa,axis=0)
    return cgfa    

def deflen(gfa):
    dist_arr = np.sqrt(np.sum(gfa**2,axis=1))
    coordA = gfa[np.argmax(dist_arr), :]
    gfa_moved = gfa[:] - coordA[:]
    dist_arr = np.sqrt(np.sum(gfa_moved**2,axis=1))
    coordB = gfa[np.argmax(dist_arr), :]
    deflen = np.linalg.norm(coordA - coordB)
    return deflen

def PolygonArea(xy):
    x = xy[:,0]
    y = xy[:,1]
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)

def defarea(gfa):
    hull = ConvexHull(gfa)
    hull_pts = gfa[hull.vertices,:]
    defarea = PolygonArea(hull_pts)
    return defarea

def multiproc(df, sid):
    xy = df.loc[df['WAFER']==sid, ['WAFER_X','WAFER_Y']].to_numpy()
    GFA_ID_o = np.concatenate(df.loc[df['WAFER']==sid,['GFA ID']].to_numpy())
    GFA_ID_noZ = np.unique(GFA_ID_o)
    GFA_ID_noZ = GFA_ID_noZ[GFA_ID_noZ != 0]
    GFA_Centroid = np.full((len(xy),2),np.nan)
    AREA = np.full((len(xy),1),np.nan)
    LEN = np.full((len(xy),1),np.nan)
    if GFA_ID_noZ != 0:
        for i in GFA_ID_noZ:
            index = GFA_ID_o == i
            gfa = xy[index]
            GFA_Centroid[index] = centr(gfa)
            LEN[index] = deflen(gfa)
            AREA[index] = defarea(gfa)
    GFA_tot = np.column_stack((GFA_Centroid, LEN, AREA)) #GFA_Class
    return GFA_tot

def main_p(df):
    uniqScrID = df['WAFER'].unique() #Use ScribeID
    multiproc1 = partial(multiproc, df)
    p = Pool(processes = 8)
    GFA_tot = np.concatenate(np.asarray(p.map(multiproc1, uniqScrID)))
    p.close()
    p.join()

    df['GFA_CentX'] = GFA_tot[:,0]
    df['GFA_CentY'] = GFA_tot[:,1]
    df['LEN'] = GFA_tot[:,2]
    df['AREA'] = GFA_tot[:,3]
    
    #Compiling Data
    scrb_GFAID = df['WAFER'].astype(str) + ' ' + df['GFA ID'].astype(str) #Use ScribeID
    _, indices = np.unique(scrb_GFAID,return_index = True)
    KNN_Training_Data = df.iloc[indices,[0, 1, 4, 5, 6, 7, 8, 9]]
    KNN_Training_Data = KNN_Training_Data.reset_index(drop=True)   

    return KNN_Training_Data

if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    df = pd.read_csv('training_set.csv',index_col=False,
                     usecols = ['LOT','WAFER','WAFER_X','WAFER_Y','GFA ID', 'GFA Class'])
    KNN_Training_Data = main_p(df)
    KNN_Training_Data.to_csv('KNN_Training_Data.csv')