# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:17:50 2019

@author: iloo
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from colorama import Fore, Style

def comp(df,df1,dfraw):
    df = df.loc[df['GFA_Class'] !=0, :]
    df1 = df1.loc[df1['GFA_Class'] != 0, :]
    comp = df['GFA_Class'] != df1['GFA_Class']
    per = np.mean(1-comp)*100
    print('Percent that are Equal: %.2f' %(per))
    df_both = df.loc[comp,:]
    scrb_GFAID = df_both['LOT_ID'].astype(str) + df_both['WAFER_ID'].astype(str)+ df_both['GFA_ID'].astype(str)
    scrb_GFAID1 = dfraw['LOT_ID'].astype(str) + dfraw['WAFER_ID'].astype(str) +dfraw['GFA_ID'].astype(str)
    dfraw_sub = dfraw.loc[scrb_GFAID1.isin(scrb_GFAID), :]
    uniqScrID = dfraw_sub['WAFER_SCRIBE_ID'].unique() 
    for p,sid in enumerate(uniqScrID): 
        xy = dfraw_sub.loc[dfraw_sub['WAFER_SCRIBE_ID']==sid, ['REAL_WAFER_X','REAL_WAFER_Y']].to_numpy()
        gfaID = dfraw_sub.loc[dfraw_sub['WAFER_SCRIBE_ID']==sid,'GFA_ID']
        uniqgfaID = gfaID.unique()
        for i,gid in enumerate(uniqgfaID):
            index = gfaID == gid
            gfa = xy[index]
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
            print(Fore.RED + 'To Exit: 1111' + Style.RESET_ALL)
            while True:
                try:
                    man_clas = int(input('Move On? '))
                except ValueError:
                    print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                    continue
                if man_clas == 1111:
                    sys.exit()
                elif man_clas != 1:
                    print(Fore.RED + 'Please Enter a Valid Value.' + Style.RESET_ALL)
                    continue
                else:
                    break                            
    return dfraw_sub,df_both

def comp1(df,df1):
    df = df.loc[df['GFA_Class'] !=0, :]
    df1 = df1.loc[df1['GFA_Class'] != 0, :]
    comp = df['GFA_Class'] != df1['GFA_Class']
    df = df.loc[comp,:]
    df1 = df1.loc[comp,:]   
    df_both1 = pd.concat([df[['LOT_ID','WAFER_ID']],df['GFA_Class'],df1['GFA_Class']],axis=1)
    df_both1.columns = ['LOT_ID','WAFER_ID','Issac_Class','Ope_Class']
    df_both1.boxplot(column=['Issac_Class','Ope_Class'],figsize = (10,8))
    df_both1.hist(column=['Issac_Class','Ope_Class'],figsize = (10,8))
    
    return df_both1
if __name__ == '__main__':
    os.chdir('GFAP CSV Files')
    df = pd.read_csv('TrainingData.csv')
    df1 = pd.read_csv('TrainingData1.csv')
    dfraw = pd.read_csv('TrainingData_Raw.csv')
#    dfraw_sub,df_both = comp(df,df1,dfraw)
    df_both1 = comp1(df,df1)