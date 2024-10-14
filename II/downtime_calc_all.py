#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:26:54 2024

@author: pouriak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, genextreme, norm, expon, gamma, beta, kstest, ks_2samp, t
#import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'
import os
import seaborn as sns
#from fitter import Fitter, get_common_distributions, get_distributions
#from sklearn.mixture import GaussianMixture
import distfit
#import statsmodels.api as sm
from scipy import stats 
#import random
import downtime_prob_modeling as fnc


Archs = ['S8H14SEAWBPG2','S12H14SEAWBPG2','S16H14SEAWBPG2','S20H14SEAWBPG2','S24H14SEAWBPG2']
#Archs = ['S20H14SEAWBPG2','S16H14SEAWBPG2','S24H14SEAWBPG2','S12H14SEAWBPG2']
#Archs = ['S16H14SEAWBPG2']
Hazard = ['100', '475', '975', '2475', '4975']
#Hazard = ['100']
input_data = pd.read_csv('input_data.csv') #change this

SA_avg=[]
SA_avg2=np.zeros((5,5))
SA_Tn=np.zeros((5,5))
for i in range(len(Archs)):
    for j in range(len(Hazard)):
        IM_arch = input_data[input_data['Arch']==Archs[i]]
        IM_arch_haz = IM_arch[IM_arch['Hazard Level']==int(Hazard[j])]
        SA_avg.append(stats.gmean(IM_arch_haz['SA_avg']))
        SA_avg2[i,j] = stats.gmean(IM_arch_haz['SA_avg'])
        SA_Tn[i,j] = np.mean(IM_arch_haz['SA_Tn'])


#bld = '/S12H14SEAWBPG2'
nshm = '/2014/'

cd=os.getcwd()
Prob_exc_SiP=[]
Prob_exc_FR=[]
Prob_irrep=[]
for k in range(len(Archs)):
    arch = '/'+Archs[k]+nshm
    for n in range(len(Hazard)): #hazard level 
        haz = '/'+Hazard[n]+'/'
        Prob_irrep.append(1-(pd.read_csv(cd+arch+haz+'DL_summary_stats.csv',index_col=None)['reconstruction/cost_impractical'][0])/2000)
        Prob_exc_SiP.append(pd.read_csv(cd+arch+haz+'outputs/'+'RS_stats.csv')['prob (RS not achieved)'][2])
        Prob_exc_FR.append(pd.read_csv(cd+arch+haz+'outputs/'+'RS_stats.csv')['prob (RS not achieved)'][0])
        #Prob_exc_SiP.append(pd.read_csv(cd+arch+haz+'RS_stats.csv')['prob (RS not achieved)'][2])
        #Prob_exc_FR.append(pd.read_csv(cd+arch+haz+'RS_stats.csv')['prob (RS not achieved)'][0])
#IM = [0.03, 0.10, 0.15, 0.25, 0.33]
#ret_period = [100, 475, 975, 2475, 4975]

#damage states
Prob_exc_DS1=[]
Prob_exc_DS2=[]
Prob_exc_DS3=[]
edp=pd.read_csv('RCSW_EDPs_REF.csv',index_col=None)

GM_haz=['2014_CS_100_ROTD50','2014_CS_475_ROTD50','2014_CS_975_ROTD50','2014_CS_2475_ROTD50','2014_CS_4975_ROTD50']
for k in Archs:
    edp_arch=edp[edp['ArchetypeActual']==k]
    for i in GM_haz:
        edp_arch_haz=edp_arch[edp_arch['GMSet']==i]
        Prob_exc_DS1.append(len(edp_arch_haz[edp_arch_haz['MaxInterStoryDrift']>=0.01])/len(edp_arch_haz))
        Prob_exc_DS2.append(len(edp_arch_haz[edp_arch_haz['MaxInterStoryDrift']>=0.022])/len(edp_arch_haz))
        Prob_exc_DS3.append(len(edp_arch_haz[edp_arch_haz['MaxInterStoryDrift']>=0.026])/len(edp_arch_haz))

# med_irrep_24=0.2977; med_irrep_12=0.275; med_irrep_16=0.275; med_irrep_20=0.2835; med_irrep_8=0.2862
# disp_irrep_24=0.2076; disp_irrep_12=0.288; disp_irrep_16=0.22; disp_irrep_20=0.2592; disp_irrep_8=0.3157

# med_sip_24=0.1214; med_sip_12=0.1252; med_sip_16=0.1252; med_sip_20=0.1174; med_sip_8=0.1330
# disp_sip_24=0.203; disp_sip_12=0.275; disp_sip_16=0.23; disp_sip_20=0.1992; disp_sip_8=0.2510

# med_ds1_24=0.153; med_ds1_12=0.1266; med_ds1_16=0.135; med_ds1_20=0.1340; med_ds1_8=0.116
# disp_ds1_24=0.2885; disp_ds1_12=0.299; disp_ds1_16=0.299; disp_ds1_20=0.2773; disp_ds1_8=0.2

# med_ds2_24=0.2692; med_ds2_12=0.2363; med_ds2_16=0.2377; med_ds2_20=0.2535; med_ds2_8=0.2468
# disp_ds2_24=0.2668; disp_ds2_12=0.2502; disp_ds2_16=0.2078; disp_ds2_20=0.2495; disp_ds2_8=0.2321

med_fr_24=0.0208; med_fr_12=0.024; med_fr=0.018
disp_fr_24=0.1344; disp_fr_12=0.1344; disp_fr=0.2

med_ds2=0.1648; med_ds1=0.0934; med_sip=0.0877; med_irrep=0.1854
disp_ds2=0.2152; disp_ds1=0.2791; disp_sip=0.2427; disp_irrep=0.2444

med_ds2_s50=0.2738; med_ds1_s50=0.129; med_sip_s50=0.1136; med_irrep_s50=0.33; med_fr_s50=0.022
disp_ds2_s50=0.4111; disp_ds1_s50=0.3032; disp_sip_s50=0.2378; disp_irrep_s50=0.4533; disp_fr_s50=0.2

med_ds2_dl=0.2329; med_ds1_dl=0.1138; med_sip_dl=0.108; med_irrep_dl=0.2583; med_fr_dl=0.02
disp_ds2_dl=0.3251; disp_ds1_dl=0.2419; disp_sip_dl=0.2984; disp_irrep_dl=0.34; disp_fr_dl=0.35

x=np.linspace(0.01,.3,1000)
y1=lognorm.cdf(x,disp_irrep,0,med_irrep)
y2=lognorm.cdf(x,disp_sip,0,med_sip)
y22=lognorm.cdf(x,disp_ds1,0,med_ds1)
y23=lognorm.cdf(x,disp_ds2,0,med_ds2)
y3=lognorm.cdf(x,disp_fr,0,med_fr)

plt.figure(figsize=(4,2.5), dpi=600)
plt.plot(x,y3, c='#8F993E', label='FR')
plt.plot(x,y2, c='#002855', label='SiP')
plt.plot(x,y22,'--', c='#0097A9', label='DS1')
plt.plot(x,y23,'--', c='#EA7600', label='DS2')
plt.plot(x,y1, c='#93272C', label='IrD')


plt.scatter(SA_avg, Prob_irrep, s=20, c='#93272C', marker='o')
plt.scatter(SA_avg, Prob_exc_SiP, s=20, c='#002855', marker='o')
plt.scatter(SA_avg, Prob_exc_DS1, s=20, c='#0097A9', marker='o')
plt.scatter(SA_avg, Prob_exc_DS2, s=20, c='#EA7600', marker='o')
#plt.scatter(IM, Prob_exc_DS3, s=20, c='purple', marker='o')
plt.scatter(SA_avg, Prob_exc_FR, s=20, c='#8F993E', marker='o')


plt.xlim(0, 0.3)
plt.ylim(-0.05, 1.05)
plt.ylabel('Probability of Exceedance', fontsize=10)
#plt.ylabel('P(RS>x|IM)', fontsize=10)
#plt.xlabel('$SA_{T=2.7 s}$ (g)', fontsize=10)
#plt.xlabel('IM', fontsize=10)
plt.xlabel('AvgSA [g]', fontsize=10)
#plt.title('Archs with 50% Increased Strength (S50%)', fontsize=10)
#plt.title('Archs with 1.25% Drift Limit (DL1.25%)', fontsize=10)
#plt.xticks(IM, ret_period)

handles, labels = plt.gca().get_legend_handles_labels()

new_order = [0, 1, 4,2, 3]  # Change the order as needed

# Create a new legend with the specified order
plt.legend([handles[i] for i in new_order], [labels[i] for i in new_order],fontsize=9,loc='lower right')


#%% LS functions comparison
plt.figure(figsize=(3.5,2), dpi=600)
plt.plot(x,lognorm.cdf(x,disp_fr,0,med_fr), c='#8F993E', label='FR (REF)')
plt.plot(x,lognorm.cdf(x,disp_fr_s50,0,med_fr_s50), c='#8F993E', linestyle='--', label='FR (S50%)')
plt.plot(x,lognorm.cdf(x,disp_fr_dl,0,med_fr_dl), c='#8F993E', linestyle=':', label='FR (DL1.25%)')

plt.plot(x,lognorm.cdf(x,disp_sip,0,med_sip), c='#002855', label='SiP (REF)')
plt.plot(x,lognorm.cdf(x,disp_sip_s50,0,med_sip_s50), c='#002855', linestyle='--', label='SiP (S50%)')
plt.plot(x,lognorm.cdf(x,disp_sip_dl,0,med_sip_dl), c='#002855', linestyle=':', label='SiP (DL1.25%)')

plt.plot(x,lognorm.cdf(x,disp_irrep,0,med_irrep), c='#93272C', label='IrD (REF)')
plt.plot(x,lognorm.cdf(x,disp_irrep_s50,0,med_irrep_s50), c='#93272C', linestyle='--', label='IrD (S50%)')
plt.plot(x,lognorm.cdf(x,disp_irrep_dl,0,med_irrep_dl), c='#93272C', linestyle=':', label='IrD (DL1.25%)')


plt.xlim(0, 0.3)
plt.ylim(-0.05, 1.05)
plt.ylabel('Probability of Exceedance', fontsize=10)
#plt.xlabel('$SA_{T=2.7 s}$ (g)', fontsize=10)
plt.xlabel('$SA_{avg}$ [g]', fontsize=10)
#plt.title('Archs with 50% Increased Strength (S50%)', fontsize=10)
#plt.title('Archs with 1.25% Drift Limit (DL1.25%)', fontsize=10)
#plt.xticks(IM, ret_period)

plt.legend(fontsize=8,loc=(-0.2, 1.05),ncol=3)



#%%
#train
Archs = ['S24H14SEAWBPG2','S8H14SEAWBPG2','S16H14SEAWBPG2','S20H14SEAWBPG2']
#Archs = ['S24H14SEAWBPG2']
#Hazard = ['4975']
Hazard = ['100', '475', '975', '2475', '4975']
#SDR=[]

#test
#n_test = 16 #test archetype
#n_haz=2 #hazard level

#prob_cdf_fr = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_fr,0,med_fr)
#prob_cdf_sip = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_sip,0,med_sip)
#prob_cdf_irrep = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_irrep,0,med_irrep)

#prob_only_sip = np.round((prob_cdf_sip-prob_cdf_irrep)*10000)
#prob_only_fr = np.round((prob_cdf_fr-prob_cdf_irrep)*10000)
#prob_irrep = np.round(prob_cdf_irrep*10000)
#prob_no_damage_sip = np.full(5, 10000) - prob_only_sip - prob_irrep
#prob_no_damage_fr = np.full(5, 10000) - prob_only_fr - prob_irrep


Prob_irrep_haz=[]
for k in range(len(Archs)):
    arch = '/'+Archs[k]+nshm
    for n in range(len(Hazard)): #hazard level 
        haz = '/'+Hazard[n]+'/'
        Prob_irrep_haz.append(1-(pd.read_csv(cd+arch+haz+'DL_summary_stats.csv',index_col=None)['reconstruction/cost_impractical'][0])/2000)
        #sdr1=pd.read_csv(cd+arch+haz+'EDP_.csv',index_col=None)['1-PID-24-1']
        #sdr2=pd.read_csv(cd+arch+haz+'EDP_.csv',index_col=None)['1-PID-24-2']
        #SDR.append(np.maximum(sdr1, sdr2))
#SDR_matrix = np.squeeze(SDR).T


DT_SiP0=[]
Rep_time_SiP_all0=[]
DT_FR0=[]
Rep_time_FR_all0=[]

for k in range(len(Archs)):
    arch = '/'+Archs[k]+nshm
    for n in range(len(Hazard)): #hazard level
        haz = '/'+Hazard[n]+'/'
    
    #shelter-in-place
        Rep_time_SiP_arr=pd.read_excel(cd+arch+haz+'outputs/'+'RT_stepfunc_SiP.xlsx',engine='openpyxl',sheet_name=['RSeq1','RSeq2','RSeq3','RSeq4','RSeq5','RSeq6','RSeq7'])
        DT_SiP0.append(pd.read_csv(cd+arch+haz+'outputs/'+'DT_stepfunc_SiP.csv',index_col=None,header=None).iloc[2:2+round(2000*(1-Prob_irrep_haz[k])),-1])
        
        Rep_time_SiP_all0.append(np.maximum.reduce([Rep_time_SiP_arr['RSeq1'].iloc[:,-1],Rep_time_SiP_arr['RSeq2'].iloc[:,-1],Rep_time_SiP_arr['RSeq3'].iloc[:,-1],
                             Rep_time_SiP_arr['RSeq4'].iloc[:,-1],Rep_time_SiP_arr['RSeq5'].iloc[:,-1],Rep_time_SiP_arr['RSeq6'].iloc[:,-1],
                             Rep_time_SiP_arr['RSeq7'].iloc[:,-1]]))
    
    #functional recovery
        Rep_time_FR_arr=pd.read_excel(cd+arch+haz+'outputs/'+'RT_stepfunc_FR.xlsx',engine='openpyxl',sheet_name=['RSeq1','RSeq2','RSeq3','RSeq4','RSeq5','RSeq6','RSeq7'])
        DT_FR0.append(pd.read_csv(cd+arch+haz+'outputs/'+'DT_stepfunc_FR.csv',index_col=None,header=None).iloc[2:2+round(2000*(1-Prob_irrep_haz[k])),-1])
        
        Rep_time_FR_all0.append(np.maximum.reduce([Rep_time_FR_arr['RSeq1'].iloc[:,-1],Rep_time_FR_arr['RSeq2'].iloc[:,-1],Rep_time_FR_arr['RSeq3'].iloc[:,-1],
                             Rep_time_FR_arr['RSeq4'].iloc[:,-1],Rep_time_FR_arr['RSeq5'].iloc[:,-1],Rep_time_FR_arr['RSeq6'].iloc[:,-1],
                             Rep_time_FR_arr['RSeq7'].iloc[:,-1]]))
    
    
DT_SiP=pd.concat(DT_SiP0).reset_index(drop=True)
Rep_time_SiP_all=np.concatenate(Rep_time_SiP_all0)

DT_FR=pd.concat(DT_FR0).reset_index(drop=True)
Rep_time_FR_all=np.concatenate(Rep_time_FR_all0)

#DT_SiP_stats_actual = [np.percentile(DT_SiP,5), np.mean(DT_SiP), np.median(DT_SiP), np.percentile(DT_SiP,95)]

#DT_FR_stats_actual = [np.percentile(DT_FR,5), np.mean(DT_FR), np.median(DT_FR), np.percentile(DT_FR,95)]

#DT_FR=pd.concat([DT_FR0[0],DT_FR0[1],DT_FR0[2],DT_FR0[3],DT_FR0[4]]).reset_index(drop=True)
#Rep_time_FR_all=np.concatenate((Rep_time_FR_all0[0],Rep_time_FR_all0[1],Rep_time_FR_all0[2],Rep_time_FR_all0[3],Rep_time_FR_all0[4]))

#%%

#Sip calcs begin here

Rep_time_SiP = Rep_time_SiP_all[Rep_time_SiP_all != 0] #repair time
indx_zero = np.where(Rep_time_SiP_all==0)[0] #no repair needed; DT=inspection time


IF_SiP_all = DT_SiP - Rep_time_SiP_all
IF_SiP = IF_SiP_all.drop(indx_zero).reset_index(drop=True)  #impeding factor delay

x_sip=np.linspace(0.001,max(Rep_time_SiP),len(Rep_time_SiP))
#x_sip=np.linspace(0.001,.999,1000)

plt.figure(figsize=(3.5,2.5), dpi=600)
#ax=sns.histplot(fnc.Min_Max_Normalization(Rep_time_SiP), stat="density", bins=20, kde=False, kde_kws={"bw_adjust":1}, color='gray')
ax=sns.histplot(Rep_time_SiP, stat="density", bins=24, kde=False, kde_kws={"bw_adjust":1}, color='gray')

#ax.lines[0].set_color('crimson')
#params_beta_rep = beta.fit(fnc.Min_Max_Normalization(Rep_time_SiP),floc=0,fscale=1)
params_beta_rep = beta.fit(Rep_time_SiP)

RTmax = Rep_time_SiP.max()
RTmin = Rep_time_SiP.min()
    
#RT_SiP_sampled = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=int(prob_only_sip[n_haz]))) + RTmin
RT_SiP_dist = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=10000)) + RTmin


plt.plot(x_sip,beta.pdf(x_sip, params_beta_rep[0], params_beta_rep[1], params_beta_rep[2], params_beta_rep[3]),color='#AC145A',label='Beta fit')
#plt.plot(x_sip,genextreme.pdf(x_sip, params_genextreme_rep[0], params_genextreme_rep[1], params_genextreme_rep[2]),color='green')
#ax.lines[0].set_color('crimson')
plt.legend(fontsize=9)
plt.ylabel('Density', fontsize=10)
plt.xlabel('Repair Time - SiP [days]', fontsize=10)
plt.xlim([0,140])
plt.ylim([0,0.04])
plt.yticks([0,0.01,0.02,0.03,0.04])

dist = distfit.distfit()
#dist.fit_transform(Rep_time_SiP)
print(dist.fit_transform(Rep_time_SiP)['model']['name'])


#params_gamma = gamma.fit(IF_SiP)
#params_lognorm = lognorm.fit(IF_SiP)
params_genextreme = genextreme.fit(IF_SiP)
#params_beta = beta.fit(IF_SiP)
#params_t = t.fit(IF_SiP)

IF_SiP_sampled = genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], size=10000)

params_SiP_RT = np.array([RTmin,RTmax,params_beta_rep[0],params_beta_rep[1]])
params_SiP_IF = np.array(params_genextreme)

x_sip=np.linspace(0.1,max(IF_SiP),len(IF_SiP))
plt.figure(figsize=(3.5,2.5), dpi=600)
ax=sns.histplot(IF_SiP, stat="density", bins=80, kde=False, kde_kws={"bw_adjust":1},color='gray')
#ax=sns.histplot(IF_SiP_sampled, stat="density", bins=50, kde=False, kde_kws={"bw_adjust":1})

#ax.lines[0].set_color('crimson')
#plt.plot(x_sip,beta.pdf(x_sip, params_beta[0], params_beta[1], params_beta[2], params_beta[3]),color='orange')
plt.plot(x_sip,genextreme.pdf(x_sip, params_genextreme[0], params_genextreme[1], params_genextreme[2]),color='#D50032', label='GEV fit')
#ax.lines[0].set_color('crimson')
plt.legend(fontsize=9)
plt.ylabel('Density', fontsize=10)
plt.xlabel('IF delays - SiP [days]', fontsize=10)
plt.xlim([0,500])
plt.ylim([0,0.012])


dist = distfit.distfit()
#dist.fit_transform(IF_SiP)
print(dist.fit_transform(IF_SiP)['model']['name'])

#print(kstest(IF_SiP, 'genextreme', args=params_genextreme))
#print(kstest(IF_SiP, 'beta', args=params_beta))
#print(ks_2samp(IF_SiP,beta.rvs(params_beta[0], params_beta[1], params_beta[2], params_beta[3], len(IF_SiP))))
#print(ks_2samp(IF_SiP,genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], len(IF_SiP))))

#%%

#FR calcs begin here

Rep_time_FR = Rep_time_FR_all[Rep_time_FR_all != 0] #repair time
indx_zero = np.where(Rep_time_FR_all==0)[0]


IF_FR_all = DT_FR - Rep_time_FR_all
IF_FR = IF_FR_all.drop(indx_zero).reset_index(drop=True)  #impeding factor delay

x_fr=np.linspace(0.01,max(Rep_time_FR),len(Rep_time_FR))
#x_fr=np.linspace(0.001,0.999,10000)
plt.figure(figsize=(3.5,2.5), dpi=600)
#ax=sns.histplot(fnc.Min_Max_Normalization(Rep_time_FR), stat="density", bins=30, kde=False, kde_kws={"bw_adjust":1})
ax=sns.histplot(Rep_time_FR, stat="density", bins=24, kde=False, kde_kws={"bw_adjust":1},color='gray')

#ax.lines[0].set_color('crimson')

#params_beta_rep = beta.fit(fnc.Min_Max_Normalization(Rep_time_FR),floc=0,fscale=1)
params_beta_rep = beta.fit(Rep_time_FR)

RTmax = Rep_time_FR.max()
RTmin = Rep_time_FR.min()
    
#RT_SiP_sampled = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=int(prob_only_sip[n_haz]))) + RTmin
RT_FR_dist = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=10000)) + RTmin


plt.plot(x_fr,beta.pdf(x_fr, params_beta_rep[0], params_beta_rep[1], params_beta_rep[2], params_beta_rep[3]),color='#AC145A',label='Beta fit')
#plt.plot(x_fr,genextreme.pdf(x_fr, params_genextreme_rep[0], params_genextreme_rep[1], params_genextreme_rep[2]),color='green')
#ax.lines[0].set_color('crimson')
plt.ylabel('Density', fontsize=10)
plt.xlabel('Repair Time - FR [days]', fontsize=10)
plt.legend(fontsize=9)

plt.xlim([0,250])
plt.ylim([0,0.02])

#dist = distfit.distfit()
#dist.fit_transform(Rep_time_FR)


#params_gamma = gamma.fit(IF_FR)
#params_lognorm = lognorm.fit(IF_FR)
params_genextreme = genextreme.fit(IF_FR)
#params_beta = beta.fit(IF_FR)

IF_FR_sampled = genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], size=10000)

params_FR_RT = np.array([RTmin,RTmax,params_beta_rep[0],params_beta_rep[1]])
params_FR_IF = np.array(params_genextreme)

x_fr=np.linspace(0.1,max(IF_FR),len(IF_FR))
plt.figure(figsize=(3.5,2.5), dpi=600)
ax=sns.histplot(IF_FR, stat="density", bins=70, color='gray')
#ax.lines[0].set_color('crimson')
#plt.plot(x_fr,beta.pdf(x_fr, params_beta[0], params_beta[1], params_beta[2], params_beta[3]),color='orange')
plt.plot(x_fr,genextreme.pdf(x_fr, params_genextreme[0], params_genextreme[1], params_genextreme[2]),color='#D50032', label='GEV fit')
#plt.legend({'kde'})
#ax.lines[0].set_color('crimson')
plt.ylabel('Density', fontsize=10)
plt.xlabel('IF delays - FR [days]', fontsize=10)
plt.legend(fontsize=9)
plt.xlim([0,600])

plt.ylim([0,0.008])
#kde = sns.kdeplot(IF_FR, shade=False)
#ks_statistic, p_value = stats.kstest(IF_FR, kde.get_lines()[0].get_data()[1])
#print("KS Statistic:", ks_statistic)
#print("p-value:", p_value)

#dist = distfit.distfit()
#dist.fit_transform(IF_FR)

#print(kstest(IF_FR, 'genextreme', args=params_genextreme))
#print(kstest(IF_FR, 'beta', args=params_beta))
#print(ks_2samp(IF_FR,beta.rvs(params_beta[0], params_beta[1], params_beta[2], params_beta[3], len(IF_FR))))
#print(ks_2samp(IF_FR,genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], len(IF_FR))))








