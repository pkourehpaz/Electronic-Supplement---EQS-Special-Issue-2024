#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:05:13 2024

@author: pouriak
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, genextreme, norm, expon, gamma, beta, kstest, ks_2samp
#import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
import os

 

#% find M9 station coordinates - initial guess

#from itertools import product

# Define ranges for x and y
#lat_m9 = np.arange(49.12407, 49.35795,0.009)  # Example: 1, 2, 3
#lon_m9 = np.arange(-123.25866, -122.84852, 0.01367)  # Example: 4, 5, 6

# Create pairs of (x, y) for each combination of x and y values
#m9_coord = list(product(lat_m9, lon_m9))

#%% find M9 station coordinates - final
cd=os.getcwd()
parent_directory = cd+"/M9_GM_RS"

# List all directories within the specified path
folder_names = [name for name in os.listdir(parent_directory)
                if os.path.isdir(os.path.join(parent_directory, name))]

#print(folder_names)

import re
def extract_numbers(s):
    # This regular expression matches both integers and floats
    numbers = re.findall(r'-?\b\d+\.?\d*\b', s)
    # Convert found numbers to float if '.' in them, else to int
    return [float(num) if '.' in num else int(num) for num in numbers]

m9_coord_rev=[]
for i  in range(len(folder_names)):
    m9_coord_rev.append(extract_numbers(folder_names[i]))

m9_coord_rev_num = np.squeeze(m9_coord_rev)



#%%

from scipy import spatial
#from selenium import webdriver
#import time
#from selenium.webdriver.chrome.service import Service
cd=os.getcwd()

blds = pd.read_excel(cd+"/Inventory/EmporisData_PK_ModernC2H.xlsx", sheet_name='buildings')
blds_h = blds[blds['Year']>=2005]
blds_8 = blds_h[blds_h['Floors']>=8]
blds_8_24 = blds_8[blds_8['Floors']<=24]


lat=blds_8_24.iloc[:,-2].reset_index(drop=True)
lon=blds_8_24.iloc[:,-1].reset_index(drop=True)
blds_location=[]

#%%
m9_coord_s0=[]
for i in range(len(blds_8_24)):
    m9_coord_s0.append(m9_coord_rev[spatial.KDTree(m9_coord_rev).query(blds_location[i])[1]])

#m9_coord_s = list(dict.fromkeys(m9_coord_s0))
aaa=np.squeeze(blds_location)
m9_coord_s0_array=np.squeeze(m9_coord_s0)
#richmond
indx_richmond = (np.where((aaa[:, 0] > 49.12) & (aaa[:, 0] < 49.197) & (aaa[:, 1] > -123.19) & (aaa[:, 1] < -122.96))[0])
#indx_richmond = np.concatenate((indx_richmond1,np.array([217,218,219])))

#vancouver
indx_van = (np.where((aaa[:, 0] > 49.208) & (aaa[:, 0] < 49.294) & (aaa[:, 1] > -123.256) & (aaa[:, 1] < -123.03))[0])

#north vancouver
indx_norvan = (np.where((aaa[:, 0] > 49.309) & (aaa[:, 0] < 49.348) & (aaa[:, 1] > -123.209) & (aaa[:, 1] < -123.07))[0])
#indx_norvan = np.concatenate((indx_norvan1,np.array([0,1,2])))

#burnaby
indx_burnaby = (np.where((aaa[:, 0] > 49.19) & (aaa[:, 0] < 49.281) & (aaa[:, 1] > -123.03) & (aaa[:, 1] < -122.85))[0])
#indx_burnaby = np.concatenate((indx_burnaby1,np.array([171,174])))

print(len(indx_richmond)+len(indx_van)+len(indx_norvan)+len(indx_burnaby))

bld_richmond = aaa[indx_richmond]
bld_van = aaa[indx_van]
bld_norvan = aaa[indx_norvan]
bld_burnaby = aaa[indx_burnaby]


#%% calculate SA_avg & SA_T1 values for M9 records at all sites
IM = [0.023, 0.071,	0.107, 0.166, 0.213]

from scipy.stats.mstats import gmean
cd=os.getcwd()

k=len(m9_coord_s0)
#k=5
SA_avg_all = [[] for _ in range(k)]
SA_T1_all = [[] for _ in range(k)]
for i in range(k):
    lat=str(m9_coord_s0[i][0])
    lon=str(m9_coord_s0[i][1])
    m9_direct = cd+"/M9_GM_RS/"+"("+lat+","+lon+")/"

    RS_60 = pd.read_excel(m9_direct+"RS_UBC"+"("+lat+","+lon+").xlsx").iloc[:,1:61]
    RS_30 =[]
 
    for n in np.arange(0,60,2):
        RS_30.append(np.sqrt(RS_60.iloc[:,n] * RS_60.iloc[:,n+1]))
    
    for m in range(30):
        SA_avg_all[i].append(gmean(RS_30[m][71:700]))
        SA_T1_all[i].append(RS_30[m][100])

#%% hazus parameters

beta_DS = 0.441
theta_DS = [0.5577, 1.1398, 1.7305, 2.2368]

def RecT_Hazus(SA_T1):
    p_ds = lognorm.cdf(SA_T1,beta_DS,0,theta_DS)
    recovery_time = 10*(p_ds[0]-p_ds[1]) + 120*(p_ds[1]-p_ds[2]) + 480*(p_ds[2]-p_ds[3]) + 960*(p_ds[3])
    return recovery_time

#%% distribution parameters estimation
p = pd.read_excel('results.xlsx', sheet_name='parameters')

SiP_IF_genxtreme=[p.iloc[[1,5,9,13,17],1].reset_index(drop=True), p.iloc[[2,6,10,14,18],1].reset_index(drop=True), p.iloc[[3,7,11,15,19],1].reset_index(drop=True)]
FR_IF_genxtreme=[p.iloc[[1,5,9,13,17],3].reset_index(drop=True), p.iloc[[2,6,10,14,18],3].reset_index(drop=True), p.iloc[[3,7,11,15,19],3].reset_index(drop=True)]


SiP_RT_beta=[p.iloc[[1,5,9,13,17],2].reset_index(drop=True), p.iloc[[2,6,10,14,18],2].reset_index(drop=True), p.iloc[[3,7,11,15,19],2].reset_index(drop=True), p.iloc[[4,8,12,16,20],2].reset_index(drop=True)]
FR_RT_beta=[p.iloc[[1,5,9,13,17],4].reset_index(drop=True), p.iloc[[2,6,10,14,18],4].reset_index(drop=True), p.iloc[[3,7,11,15,19],4].reset_index(drop=True), p.iloc[[4,8,12,16,20],4].reset_index(drop=True)]

def param_extrp(SA_avg,y_vector):
   
    log_x = np.log(IM)
    log_y = np.log(y_vector.tolist())
    
    log_x_interp = np.linspace(log_x.min(), log_x.max(), 100)
    log_y_interp = np.interp(log_x_interp, log_x, log_y)
    #y_interp = np.exp(log_y_interp)
    
    #desired_IM = SA_avg
    log_desired_IM = np.log(SA_avg)
    
    # Interpolate the log-transformed value to find the corresponding log y value
    log_desired_y = np.interp(log_desired_IM, log_x_interp, log_y_interp)
    
    # Convert the interpolated log y value back to the original scale
    return np.exp(log_desired_y)

#print(param_extrp(.2,SiP_RT_beta[0]))

#%%
k = len(SA_avg_all)
#k = 5
DT_sip_mean = [[] for _ in range(30)]
DT_fr_mean = [[] for _ in range(30)]
DT_hazus = [[] for _ in range(30)]

for j in range(30):
#for i in range(k):
    for i in range(k):
        SA_avg = SA_avg_all[i][j]
        SA_T1 = SA_T1_all[i][j]
        med_fr=0.018; disp_fr=0.2
        
        med_ds2=0.1648; med_ds1=0.0934; med_sip=0.0877; med_irrep=0.1854
        disp_ds2=0.2152; disp_ds1=0.2791; disp_sip=0.2427; disp_irrep=0.2444
        
        
        prob_cdf_fr = lognorm.cdf(SA_avg,disp_fr,0,med_fr)
        prob_cdf_sip = lognorm.cdf(SA_avg,disp_sip,0,med_sip)
        prob_cdf_irrep = lognorm.cdf(SA_avg,disp_irrep,0,med_irrep)
        
        prob_only_sip = np.round((prob_cdf_sip-prob_cdf_irrep)*10000)
        prob_only_fr = np.round((prob_cdf_fr-prob_cdf_irrep)*10000)
        prob_irrep = np.round(prob_cdf_irrep*10000)
        prob_no_damage_sip = 10000 - prob_only_sip - prob_irrep
        prob_no_damage_fr = 10000 - prob_only_fr - prob_irrep
        
        
        Insp_time_sip = np.random.lognormal(mean=np.log(5), sigma=0.54, size=int(prob_no_damage_sip))
        Insp_time_fr = np.random.lognormal(mean=np.log(5), sigma=0.54, size=int(prob_no_damage_fr))
        Replace_time = np.random.lognormal(mean=np.log(445), sigma=0.57, size=int(prob_irrep)) + np.full(int(prob_irrep), 14*16)
        
        
        RT_FR_sampled = (param_extrp(SA_avg,FR_RT_beta[1])-param_extrp(SA_avg,FR_RT_beta[0])) * (beta.rvs(param_extrp(SA_avg,FR_RT_beta[2]), param_extrp(SA_avg,FR_RT_beta[3]), size=int(prob_only_fr))) + param_extrp(SA_avg,FR_RT_beta[0])
        IF_FR_sampled = genextreme.rvs(-param_extrp(SA_avg, -FR_IF_genxtreme[0]), param_extrp(SA_avg, FR_IF_genxtreme[1]), param_extrp(SA_avg, FR_IF_genxtreme[2]), size=int(prob_only_fr))
        
        RT_SiP_sampled = (param_extrp(SA_avg,SiP_RT_beta[1])-param_extrp(SA_avg,SiP_RT_beta[0])) * (beta.rvs(param_extrp(SA_avg,SiP_RT_beta[2]), param_extrp(SA_avg,SiP_RT_beta[3]), size=int(prob_only_sip))) + param_extrp(SA_avg,SiP_RT_beta[0])
        IF_SiP_sampled = genextreme.rvs(-param_extrp(SA_avg, -SiP_IF_genxtreme[0]), param_extrp(SA_avg, SiP_IF_genxtreme[1]), param_extrp(SA_avg, SiP_IF_genxtreme[2]), size=int(prob_only_sip))
        
        
        DT_sip_predicted = list(Insp_time_sip)+list(Replace_time)+list(IF_SiP_sampled+RT_SiP_sampled)
        DT_fr_predicted = list(Insp_time_fr)+list(Replace_time)+list(IF_FR_sampled+RT_FR_sampled)
        DT_hazus_predicted = RecT_Hazus(SA_T1)
        
        DT_sip_mean[j].append(np.mean(DT_sip_predicted))
        DT_fr_mean[j].append(np.mean(DT_fr_predicted))
        DT_hazus[j].append(DT_hazus_predicted)

#%% plot recovery trajectories
x0=np.array([-15,0,0])
y0=np.arange(0,1+1/220,1/219)
y=np.concatenate(([1, 1, 0], y0))*100

fr_final=[]
for i in range(30):
    fr_final.append(DT_fr_mean[i][-1])

med_m9_fr_0=np.median(fr_final)
if med_m9_fr_0 in fr_final:
    median_exists = True
    med_m9_fr = med_m9_fr_0
else:
    median_exists = False
    # Find the closest value in fr_final to the median
    med_m9_fr = fr_final[np.argmin(np.abs(fr_final - med_m9_fr_0))]
    med_fr_index = np.where(np.array(fr_final) == med_m9_fr)[0]


plt.figure(figsize=(3.25,2.25), dpi=600)
for i in range(30):
    x=np.sort(np.concatenate((x0,DT_fr_mean[i])))
    
    if DT_fr_mean[i][-1]==DT_fr_mean[8][-1]: #index correspond to SiP median
        plt.step(x, y, where='mid', color='#8F993E', linewidth=1.5, label='FR')
            
    else:
        plt.step(x, y, where='mid', color='#8F993E', alpha=0.2, linewidth=1)


# shelter-in-place
sip_final=[]
for i in range(30):
    sip_final.append(DT_sip_mean[i][-1])

med_m9_sip_0=np.median(sip_final)
if med_m9_sip_0 in sip_final:
    median_exists = True
    med_m9_sip = med_m9_sip_0
else:
    median_exists = False
    # Find the closest value in sip_final to the median
    med_m9_sip = sip_final[np.argmin(np.abs(sip_final - med_m9_sip_0))]
    med_sip_index = np.where(np.array(sip_final) == med_m9_sip)[0]

for i in range(30):
    x=np.sort(np.concatenate((x0,DT_sip_mean[i])))
    
    if DT_sip_mean[i][-1]==DT_sip_mean[8][-1]: #index correspond to SiP median med_sip_index=8
        plt.step(x, y, where='mid', color='#002855', linewidth=1.5, label='SiP')
            
    else:
        plt.step(x, y, where='mid', color='#002855', alpha=0.2, linewidth=1)

# hazus
hazus_final=[]
for i in range(30):
    hazus_final.append(DT_hazus[i][-1])

med_m9_hazus_0=np.median(hazus_final)
if med_m9_hazus_0 in hazus_final:
    median_exists = True
    med_m9_hazus = med_m9_hazus_0
else:
    median_exists = False
    # Find the closest value in fr_final to the median
    med_m9_hazus = hazus_final[np.argmin(np.abs(hazus_final - med_m9_hazus_0))]
    med_hazus_index = np.where(np.array(hazus_final) == med_m9_hazus)[0]

for i in range(30):
    x=np.sort(np.concatenate((x0,DT_hazus[i])))
    
    if DT_fr_mean[i][-1]==med_m9_fr:
        plt.step(x, y, where='mid', color='darkred', linewidth=1.5, label='HAZUS')
            
    else:
        plt.step(x, y, where='mid', color='darkred', alpha=.4, linewidth=1)

plt.ylabel('% of Buildings Recovered')  
#plt.ylabel('Portfolio Recovery (%)')  
plt.xlabel('Recovery Time [days]')
#plt.xlabel('Recovery Time')
plt.xlim([-15,720])
plt.ylim([-5,105])
plt.xticks([0,100,200,300,400,500,600,700])
plt.legend(fontsize=9)

#%%
#FR recovery time disaggregation by city
m9_scn = 8
x0=np.array([-15,0,0])
plt.figure(figsize=(3.25,2.25), dpi=600)

l=len(indx_richmond)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_fr_mean[m9_scn][i] for i in indx_richmond]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, where='mid', color='#0097a9', linewidth=1.5, label='Richmond')


l=len(indx_burnaby)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_fr_mean[m9_scn][i] for i in indx_burnaby]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, where='mid', color='#500778', linewidth=1.5, label='Burnaby')


l=len(indx_van)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_fr_mean[m9_scn][i] for i in indx_van]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, where='mid', color='#93272C', linewidth=1.5, label='Vancouver')


l=len(indx_norvan)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_fr_mean[m9_scn][i] for i in indx_norvan]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, where='mid', color='#8F993E', linewidth=1.5, label='North Vancouver')

plt.xlim([-15,400])
plt.xticks([0,50,100,150,200,250,300,350,400])
plt.ylabel('% of Buildings Recovered')  
#plt.xlabel('Functional Recovery Time [days]')


#SiP recovery time disaggregation by city
m9_scn = 8
x0=np.array([-15,0,0])
#plt.figure(figsize=(3.25,2.25), dpi=600)

l=len(indx_richmond)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_sip_mean[m9_scn][i] for i in indx_richmond]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, '--',where='mid', color='#0097A9', linewidth=1.5, label='Richmond')


l=len(indx_burnaby)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_sip_mean[m9_scn][i] for i in indx_burnaby]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, '--',where='mid', color='#500778', linewidth=1.5, label='Burnbay')


l=len(indx_van)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_sip_mean[m9_scn][i] for i in indx_van]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, '--',where='mid', color='#93272C', linewidth=1.5, label='Vancouver')


l=len(indx_norvan)
y0=np.arange(0,1+1/l,1/(l-1))
y=np.concatenate(([1, 1, 0], y0))*100
DT_fr_mean_s = [DT_sip_mean[m9_scn][i] for i in indx_norvan]
x=np.sort(np.concatenate((x0,DT_fr_mean_s)))
plt.step(x, y, '--',where='mid', color='#8F993E', linewidth=1.5, label='North Vancouver')
plt.legend(loc='lower right', fontsize=9)


plt.xlim([-15,400])
plt.xticks([0,50,100,150,200,250,300,350,400])
plt.ylabel('% of Buildings Recovered')  
plt.xlabel('Recovery Time [days]')


