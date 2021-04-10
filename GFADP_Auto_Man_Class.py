# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:46:41 2019
@author: iloo
"""
import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import sys
from colorama import Fore, Style
from IPython import get_ipython

#GFA Class: 0 = None, 1 = Streak, 2 = Focus Spot, 3 = All Over 
get_ipython().run_line_magic('matplotlib', 'inline')

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

def classengine(df):
    uniqScrID = df['WAFER_SCRIBE_ID'].unique()
    df[['REAL_WAFER_X','REAL_WAFER_Y']] = df[['REAL_WAFER_X','REAL_WAFER_Y']].divide(1000)
    df['GFA_ID'] = np.full((len(df),1),0)
    df['GFA_Class'] = np.full((len(df),1),0)
    df['GFA_CentX'] = np.full((len(df),1),np.nan)
    df['GFA_CentY'] = np.full((len(df),1),np.nan)
    df['GFA_AREA'] = np.full((len(df),1),np.nan)
    df['GFA_LEN'] = np.full((len(df),1),np.nan)
    df['GFA_DIST'] = np.full((len(df),1),np.nan)
    df['GFA_NUM'] = np.full((len(df),1),np.nan)
    df['GFA_IsLin'] = np.full((len(df),1),np.nan)
    df['GFA_RadDist'] = np.full((len(df),1),np.nan)
    
    uniqScrID_len = len(uniqScrID)
    
    for p,sid in enumerate(uniqScrID): 
        xy = df.loc[df['WAFER_SCRIBE_ID']==sid, ['REAL_WAFER_X','REAL_WAFER_Y']].to_numpy()
        gfaID = cluster(xy,9,6)
        gfaID_noZ = np.unique(gfaID)
        gfaID_noZ = gfaID_noZ[gfaID_noZ != 0]
        centroid = np.full((len(xy),2),np.nan)
        area = np.full((len(xy),1),np.nan)
        leng = np.full((len(xy),1),np.nan)
        clas = np.full((len(xy),1),0)
        dist = np.full((len(xy),1),np.nan)
        num = np.full((len(xy),1),np.nan)
        islin = np.full((len(xy),1),np.nan)
        radDist = np.full((len(xy),1),np.nan)
        if gfaID_noZ.size != 0:
            for i,gid in enumerate(gfaID_noZ):
                index = gfaID == gid
                gfa = xy[index]
                gfaC = centr(gfa)
                gfaleng = gfalen(gfa)
                if gfaleng > 1:
                    plt.figure(figsize=(8,8),dpi=75)
                    circle = plt.Circle((0,0),150,color='r',fill=False)
                    fig = plt.gcf()
                    ax = fig.gca()
                    ax.add_artist(circle)
                    plt.scatter(xy[~index,0],xy[~index,1],s=5,c='red')
                    plt.scatter(gfa[:,0],gfa[:,1],s=5,c='black')
                    plt.ylim((-150,150))
                    plt.xlim((-150,150))
                    plt.draw()
                    plt.pause(0.001)
                    print('--- Possible GFA %d of %d ---' %(i+1,len(gfaID_noZ)))
                    print('--- Wafer %d of %d ---' %(p,uniqScrID_len))
                    print('0 = No GFA\n1 = Streak\n2 = Focus Spot\n3 = Burp')
                    print('--- Number of Points: %d ---' %(len(gfa)))
                    print(Fore.RED + 'To Exit: 1111' + Style.RESET_ALL)
                    
                    while True:
                        try:
                            man_clas = int(input('Enter GFA Class ID: '))
                        except ValueError:
                            print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                            continue
                        if man_clas == 1111:
                            sys.exit()
                        elif man_clas > 3 or man_clas < 0:
                            print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                            continue
                        else:
                            break                            
                    if man_clas == 0:
                        gfaID[index] = 0
                    elif man_clas != 0:
                        centroid[index] = gfaC
                        dist[index] = zone(gfaC)
                        leng[index] = gfaleng
                        area[index] = gfaarea(gfa)
                        clas[index] = man_clas
                        num[index] = len(gfa)
                        if man_clas == 1:
                            gfa_r1 = gfa[:,0].reshape(-1,1)
                            gfa_r2 = gfa[:,1].reshape(-1,1)
                            reg = LinearRegression().fit(gfa_r1,gfa_r2)
                            rscore = reg.score(gfa_r1,gfa_r2)
                            if rscore > 0.2: #Edit Value
                                islin[index] = 1
                                radDist[index] = abs(reg.intercept_)/np.sqrt(reg.coef_**2 + 1**2)
                            else:
                                islin[index] = 0
                else:
                    gfaID[index] = 0
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_ID'] = gfaID
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_Class'] = clas        
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_CentX'] = centroid[:,0]
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_CentY'] = centroid[:,1]
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_AREA'] = area
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_LEN'] = leng
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_DIST'] = dist
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_NUM'] = num
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_IsLin'] = islin
            df.loc[df['WAFER_SCRIBE_ID']==sid, 'GFA_RadDist'] = radDist
            
    #Compiling Data
    scrb_GFAID = df['WAFER_SCRIBE_ID'].astype(str) + ' ' + df['GFA_ID'].astype(str) #Use ScribeID
    _, indices = np.unique(scrb_GFAID,return_index = True)
    Training_Data = df.iloc[indices,[0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    return Training_Data 
            
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    df = pd.read_csv('Defect_DataV.csv')
    df = df.dropna(how='any')
    Training_Data = classengine(df)
    print(Fore.RED + 'To Exit: 1111' + Style.RESET_ALL)
    while True:
        try:
            Test_Data = int(input('Is this Test Data? Yes = 1/No = 0: '))
        except ValueError:
            print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
            continue
        if Test_Data == '1111':
            sys.exit()
        elif Test_Data > 1 or Test_Data < 0:
            print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
            continue
        else:
            break
    if Test_Data == 1:
        Training_Data.to_csv('TestData.csv',index=False)
        df.to_csv('TestData_Raw.csv',index=False)        
    elif Test_Data == 0:
        Training_Data.to_csv('TrainingData.csv',index=False)
        df.to_csv('TrainingData_Raw.csv',index=False)