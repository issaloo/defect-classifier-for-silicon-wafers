# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:46:41 2019
@author: iloo
"""
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import DBSCAN
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

def main(filelist):
    df = pd.DataFrame(columns=['Length','DistFromCent','Area','NumberOfPts','Class'])
    for j,filename in enumerate(filelist):
        df_temp = pd.read_csv(filename)
        xy = df_temp.to_numpy()
        gfaID = cluster(xy,9,6) #Edit after sensitizing
        gfaID_noZ = np.unique(gfaID)[1:]
        if gfaID_noZ.size != 0:
            for i,gid in enumerate(gfaID_noZ):
                index = gfaID == gid
                gfa = xy[index]
                gfaC = centr(gfa)
                gfaleng = gfalen(gfa)
                plt.figure(figsize=(8,8),dpi=75)
                circle = plt.Circle((0,0),150,color='r',fill=False)
                fig = plt.gcf()
                ax = fig.gca()
                ax.add_artist(circle)
                plt.scatter(xy[~index,0],xy[~index,1],s=5,c='k')
                plt.scatter(gfa[:,0],gfa[:,1],s=5,c='r')
                plt.ylim((-150,150))
                plt.xlim((-150,150))
                plt.draw()
                plt.pause(0.001)
                print('--- Possible GFA {} of {} ---'.format(i+1,len(gfaID_noZ)))
                print('--- File {} of {} ---' ,format(j+1,len(filelist)))
                print('0 = No GFA\n1 = Streak\n2 = Focus Spot\n3 = Burp') #Edit these
                print('--- Number of Points: {} ---'.format(len(gfa)))
                print(Fore.RED + 'To Exit: 1111' + Style.RESET_ALL)
                while True:
                    try:
                        man_clas = int(input('Enter GFA Class ID: '))
                    except ValueError:
                        print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                        continue
                    if man_clas == 1111:
                        sys.exit()
                    elif man_clas > 3 or man_clas < 0: #EDIT
                        print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                        continue
                    else:
                        break                            
                dist = zone(gfaC)
                area = gfaarea(gfa)
                num = len(gfa)
                df.append({'Length':gfaleng,'DistFromCent':dist,'Area':area,'NumberOfPts':num,'Class':man_clas})
        return df
            
if __name__ == '__main__':
    os.chdir('GFAP CSV Files') #Change Directory
    filelist = glob.glob('*.csv')
    Training_Data = main(filelist)
    Training_Data.to_csv('BRETTV_TD.csv',index=False)
    