# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:22:07 2019

@author: iloo
"""
import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

def uniSid(df, epsn, sid):
    indx = df['WAFER'] == sid
    xy = df.loc[indx, ['WAFER_X','WAFER_Y']].to_numpy()
    gfaIDD_o = df.loc[indx,['GFA ID']].to_numpy()
    dbscan = DBSCAN(eps=epsn,min_samples=3).fit(xy)
    gfaIDD = 1 + dbscan.labels_
    accgfa = np.full((len(xy),1),np.nan)
    accngfa = np.full((len(xy),1),np.nan)
    if np.isin(1, gfaIDD):
        gfa_mask = gfaIDD >= 1
        ngfa_mask = gfaIDD == 0
        accgfa[gfa_mask] = np.mean(gfaIDD[gfa_mask] == gfaIDD_o[gfa_mask])
        accngfa[ngfa_mask] = np.mean(gfaIDD[ngfa_mask] == gfaIDD_o[ngfa_mask])
    gfaACCtot = np.column_stack((accgfa,accngfa, gfaIDD))
    return gfaACCtot

def main_p(df,epsn):
    uniqScrID = df['WAFER'].unique()
    uniqScrID = uniqScrID.astype(int)
    p = Pool(processes=8)
    uniSid1 = partial(uniSid, df, epsn)
    gfaACCtot = np.concatenate(np.asarray(p.map(uniSid1,uniqScrID)))
    p.close()
    p.join()

    df['GFA_ID'] = gfaACCtot[:,2]
    df['GFA_ACC'] = gfaACCtot[:,0] 
    df['nGFA_ACC'] = gfaACCtot[:,1]
    scrb_GFAID = df['WAFER'].astype(str) + ' ' + df['GFA_ID'].astype(str) #Use ScribeID
    _, indices = np.unique(scrb_GFAID,return_index = True)
    output = df.iloc[indices,[0, 1, 7, 8, 9]]
    output1 = output.dropna()
    accgfa = np.mean(output1['GFA_ACC'])
    accngfa = np.mean(output1['nGFA_ACC'])

    return accgfa,accngfa
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    df = pd.read_csv('training_set.csv',index_col=0)
    bb = np.linspace(3,11,9)
    accgfa = np.zeros(shape = (len(bb)))
    accngfa = np.zeros(shape = (len(bb)))
    for i in bb:
        accgfa[i], accngfa[i] = main_p(df,i)

accgfa = accgfa*100
accngfa = accngfa*100
plt.plot(bb,accgfa,bb,accngfa)
plt.xlabel("eps",fontsize = 16)
plt.ylabel("Accuracy (%)", fontsize = 16)
plt.grid(which='major',axis='both')
plt.gca().legend(('GFA Data Point Accuracy', 'Non-GFA Data Point Accuracy')) 