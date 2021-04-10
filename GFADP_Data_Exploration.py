# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:05:05 2019

@author: iloo
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt

def dataE(df):
    #Remove nan
    index = (df['GFA_ID'] != 0) & (df['GFA_Class'] != 3)
    df_sub = df.loc[index,
                    ['GFA_LEN','GFA_AREA','GFA_Class']]
    #Histogram w/ outliers
    df_sub.hist(column=['GFA_LEN','GFA_AREA'],bins=50, figsize = (10,8))
    #Scatter w/ outliers
    df_substrk = df_sub.loc[df_sub['GFA_Class']==1,['GFA_LEN','GFA_AREA']]
    df_subFS = df_sub.loc[df_sub['GFA_Class']==2,['GFA_LEN','GFA_AREA']]
    plt.figure(figsize=(10,8),dpi=75)
    plt.scatter(df_substrk['GFA_LEN'],df_substrk['GFA_AREA'],
                c='black',label='Streak')
    plt.scatter(df_subFS['GFA_LEN'],df_subFS['GFA_AREA'],
                c='yellow',label='Focus Spot')
    plt.xlabel('GFA_LEN',fontsize=15)
    plt.ylabel('GFA_AREA',fontsize=15)
    plt.legend()
    plt.title('Area vs Length w/ Outliers',fontsize=15)
    #Boxplot w/outliers
#    df_sub.boxplot(column = ['GFA_LEN','GFA_AREA'], by='GFA_Class')
    
    #Remove SD outliers
    z_LEN = np.abs(stats.zscore(df_sub['GFA_LEN']))
    z_AREA = np.abs(stats.zscore(df_sub['GFA_AREA']))
    
    #Scatter w/o SD outliers
    df_subSD = df_sub.loc[(z_AREA <3)|(z_LEN<3),:]
    df_subSDstrk = df_subSD.loc[df_subSD['GFA_Class']==1,['GFA_LEN','GFA_AREA']]
    df_subSDFS = df_subSD.loc[df_subSD['GFA_Class']==2,['GFA_LEN','GFA_AREA']]
    plt.figure(figsize=(10,8),dpi=75)
    plt.scatter(df_subSDstrk['GFA_LEN'],df_subSDstrk['GFA_AREA'],
                c='black',label='Streak')
    plt.scatter(df_subSDFS['GFA_LEN'],df_subSDFS['GFA_AREA'],
                c='yellow',label='Focus Spot')
    plt.xlabel('GFA_LEN',fontsize=15)
    plt.ylabel('GFA_AREA',fontsize=15)
    plt.legend()
    plt.title('Area vs Length w/o SD Outliers',fontsize=15)
      
    #Remove IQR outliers
    q1 = df_sub[['GFA_LEN','GFA_AREA']].quantile(0.25,axis=1)
    q3 = df_sub[['GFA_LEN','GFA_AREA']].quantile(0.75,axis=1)
    iQR = q1 - q3
    
    index_len = (df_sub['GFA_LEN'] < (q3.iloc[0] + 1.5*iQR.iloc[0]))
    index_area = (df_sub['GFA_AREA'] < (q3.iloc[1] + 1.5*iQR.iloc[1]))
    df_subIQRlen = df_sub.loc[index_len, 
                              ['GFA_LEN','GFA_Class']]
    df_subIQRarea = df_sub.loc[index_area,
                              ['GFA_AREA','GFA_Class']]
    
    df_subIQRboth = df_sub.loc[index_len & index_area, 
                               ['GFA_LEN','GFA_AREA','GFA_Class']]
    
    #Boxplot w/o IQR outliers
    df_subIQRlen.boxplot(column='GFA_LEN', by='GFA_Class', figsize = (10,8))
    df_subIQRarea.boxplot(column='GFA_AREA', by='GFA_Class', figsize = (10,8))
    
    #Scatter w/o IQR outliers
#    df_subIQRboth.plot(x ='GFA_LEN',y='GFA_AREA',kind='scatter',
#                       title = 'Scatter)
    plt.figure(figsize=(10,8),dpi=75)
    df_subIQRstrk = df_subIQRboth.loc[df_subIQRboth['GFA_Class']==1,['GFA_LEN','GFA_AREA']]
    df_subIQRFS = df_subIQRboth.loc[df_subIQRboth['GFA_Class']==2,['GFA_LEN','GFA_AREA']]
    plt.figure(figsize=(10,8),dpi=75)
    plt.scatter(df_subIQRstrk['GFA_LEN'],df_subIQRstrk['GFA_AREA'],
                c='black',label='Streak')
    plt.scatter(df_subIQRFS['GFA_LEN'],df_subIQRFS['GFA_AREA'],
                c='yellow',label='Focus Spot')
    plt.xlabel('GFA_LEN',fontsize=15)
    plt.ylabel('GFA_AREA',fontsize=15)
    plt.legend()
    plt.title('Area vs Length w/o IQR Outliers',fontsize=15)
    
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    dataset = pd.read_csv('TrainingData.csv')
    dataset2 = pd.read_csv('TestData.csv')
    dataE(dataset)
    dataE(dataset2)