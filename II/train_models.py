#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:55:16 2024

@author: pouriak
"""
import numpy as np
import pandas as pd
import downtime_prob_modeling as fnc

Archs = [['S8H14SEAWBPG2','S12H14SEAWBPG2','S16H14SEAWBPG2','S20H14SEAWBPG2'],['S8H14SEAWBPG2','S12H14SEAWBPG2','S16H14SEAWBPG2','S24H14SEAWBPG2'],
         ['S8H14SEAWBPG2','S12H14SEAWBPG2','S20H14SEAWBPG2','S24H14SEAWBPG2'],['S8H14SEAWBPG2','S20H14SEAWBPG2','S16H14SEAWBPG2','S24H14SEAWBPG2'],
         ['S20H14SEAWBPG2','S12H14SEAWBPG2','S16H14SEAWBPG2','S24H14SEAWBPG2']]

n_test = [24,20,16,12]

n_t = 5 #no of cross-validation folds (archetypes)

Hazard = ['100']
n_haz=0 #hazard level
X_100=[]
for i in range(n_t): 
    X_100.append(fnc.DT_surrogate(Archs[i], Hazard, n_test[i], n_haz))

Hazard = ['475']
n_haz=1 #hazard level
X_475=[]
for i in range(n_t): 
    X_475.append(fnc.DT_surrogate(Archs[i], Hazard, n_test[i], n_haz))
    
Hazard = ['975']
n_haz=2 #hazard level
X_975=[]
for i in range(n_t): 
    X_975.append(fnc.DT_surrogate(Archs[i], Hazard, n_test[i], n_haz))
    
Hazard = ['2475']
n_haz=3 #hazard level
X_2475=[]
for i in range(n_t): 
    X_2475.append(fnc.DT_surrogate(Archs[i], Hazard, n_test[i], n_haz))
    
Hazard = ['4975']
n_haz=4 #hazard level
X_4975=[]
for i in range(n_t): 
    X_4975.append(fnc.DT_surrogate(Archs[i], Hazard, n_test[i], n_haz))
    

#%% 12-24 stories
X=X_100
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1]])
DT_SiP_100 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1]])
DT_FR_100 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_475
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1]])
DT_SiP_475 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1]])
DT_FR_475 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_975
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1]])
DT_SiP_975 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1]])
DT_FR_975 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_2475
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1]])
DT_SiP_2475 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1]])
DT_FR_2475 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_4975
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1]])
DT_SiP_4975 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1]])
DT_FR_4975 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))


#%% all archetypes
X=X_100
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0], X[4][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1], X[4][0][1]])
DT_SiP_100 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0], X[4][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1], X[4][1][1]])
DT_FR_100 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_475
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0], X[4][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1], X[4][0][1]])
DT_SiP_475 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0], X[4][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1], X[4][1][1]])
DT_FR_475 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_975
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0], X[4][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1], X[4][0][1]])
DT_SiP_975 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0], X[4][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1], X[4][1][1]])
DT_FR_975 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_2475
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0], X[4][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1], X[4][0][1]])
DT_SiP_2475 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0], X[4][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1], X[4][1][1]])
DT_FR_2475 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))

X=X_4975
DT_SiP_actual = np.squeeze([X[0][0][0], X[1][0][0], X[2][0][0], X[3][0][0], X[4][0][0]])
DT_SiP_predict = np.squeeze([X[0][0][1], X[1][0][1], X[2][0][1], X[3][0][1], X[4][0][1]])
DT_SiP_4975 = pd.DataFrame(np.concatenate((DT_SiP_actual,DT_SiP_predict),axis=1))
DT_FR_actual = np.squeeze([X[0][1][0], X[1][1][0], X[2][1][0], X[3][1][0], X[4][1][0]])
DT_FR_predict = np.squeeze([X[0][1][1], X[1][1][1], X[2][1][1], X[3][1][1], X[4][1][1]])
DT_FR_4975 = pd.DataFrame(np.concatenate((DT_FR_actual,DT_FR_predict),axis=1))










