#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:26:54 2024

@author: pouriak
"""
def Min_Max_Normalization (RT):
    normalized_data = (RT - RT.min()) / (max(RT.max() - RT.min(),0.1))
    epsilon = 1e-3  # Small buffer to ensure we don't hit 0 or 1
    return(normalized_data * (1 - 2 * epsilon) + epsilon)
    


def DT_surrogate(Archs, Hazard, n_test, n_haz):
    import numpy as np
    import pandas as pd
    #import matplotlib.pyplot as plt
    from scipy.stats import lognorm, genextreme, norm, expon, gamma, beta, kstest, ks_2samp
    #import seaborn as sns
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = "Arial"
    import os

    from scipy import stats 
    #import random
    
    Archs_all = ['S8H14SEAWBPG2','S12H14SEAWBPG2','S16H14SEAWBPG2','S20H14SEAWBPG2','S24H14SEAWBPG2'] #adjust this
    Hazard_all = ['100', '475', '975', '2475', '4975']
    #Hazard = ['100']
    input_data = pd.read_csv('input_data.csv')
    nshm = '/2014/'
    cd=os.getcwd()
    
    #SA_avg=[]
    SA_avg2=np.zeros((5,5))
    #SA_Tn=np.zeros((5,5))
    for i in range(len(Archs_all)):
        for j in range(len(Hazard_all)):
            IM_arch = input_data[input_data['Arch']==Archs_all[i]]
            IM_arch_haz = IM_arch[IM_arch['Hazard Level']==int(Hazard_all[j])]
            #SA_avg.append(stats.gmean(IM_arch_haz['SA_avg']))
            SA_avg2[i,j] = stats.gmean(IM_arch_haz['SA_avg'])
            #SA_Tn[i,j] = np.mean(IM_arch_haz['SA_Tn'])

    med_ds2=0.1648; med_ds1=0.0934; med_sip=0.0877; med_irrep=0.1854; med_fr=0.018
    disp_ds2=0.2152; disp_ds1=0.2791; disp_sip=0.2427; disp_irrep=0.2444; disp_fr=0.2
    
    
    prob_cdf_fr = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_fr,0,med_fr)
    prob_cdf_sip = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_sip,0,med_sip)
    prob_cdf_irrep = lognorm.cdf(SA_avg2[int(n_test/8-1),:],disp_irrep,0,med_irrep)
    
    prob_only_sip = np.round((prob_cdf_sip-prob_cdf_irrep)*10000)
    prob_only_fr = np.round((prob_cdf_fr-prob_cdf_irrep)*10000)
    prob_irrep = np.round(prob_cdf_irrep*10000)
    prob_no_damage_sip = np.full(5, 10000) - prob_only_sip - prob_irrep
    prob_no_damage_fr = np.full(5, 10000) - prob_only_fr - prob_irrep
    
    
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
    
    
    #Sip calcs begin here
    
    Rep_time_SiP = Rep_time_SiP_all[Rep_time_SiP_all != 0] #repair time
    indx_zero = np.where(Rep_time_SiP_all==0)[0] #no repair needed; DT=inspection time
    
    
    IF_SiP_all = DT_SiP - Rep_time_SiP_all
    IF_SiP = IF_SiP_all.drop(indx_zero).reset_index(drop=True)  #impeding factor delay
    
  
    params_beta_rep = beta.fit(Min_Max_Normalization(Rep_time_SiP),floc=0,fscale=1)
    RTmax = Rep_time_SiP.max()
    RTmin = Rep_time_SiP.min()
    
    
    RT_SiP_sampled = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=int(prob_only_sip[n_haz]))) + RTmin
    #RT_SiP_dist = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=10000)) + RTmin


    params_genextreme = genextreme.fit(IF_SiP)
    
    IF_SiP_sampled = genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], size=int(prob_only_sip[n_haz]))

    
    #FR calcs begin here
    
    Rep_time_FR = Rep_time_FR_all[Rep_time_FR_all != 0] #repair time
    indx_zero = np.where(Rep_time_FR_all==0)[0]
    
    
    IF_FR_all = DT_FR - Rep_time_FR_all
    IF_FR = IF_FR_all.drop(indx_zero).reset_index(drop=True)  #impeding factor delay
    
    params_beta_rep = beta.fit(Min_Max_Normalization(Rep_time_FR),floc=0,fscale=1)
    RTmax = Rep_time_FR.max()
    RTmin = Rep_time_FR.min()
    
    
    RT_FR_sampled = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=int(prob_only_fr[n_haz]))) + RTmin
    #RT_FR_dist = (RTmax-RTmin)*(beta.rvs(params_beta_rep[0], params_beta_rep[1], size=10000)) + RTmin
    

    params_genextreme = genextreme.fit(IF_FR)
    
    IF_FR_sampled = genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], size=int(prob_only_fr[n_haz]))
    #IF_FR_dist = genextreme.rvs(params_genextreme[0], params_genextreme[1], params_genextreme[2], size=10000)   
    
    ###
    #predictions
    Insp_time_sip = np.random.lognormal(mean=np.log(5), sigma=0.54, size=int(prob_no_damage_sip[n_haz]))
    Insp_time_fr = np.random.lognormal(mean=np.log(5), sigma=0.54, size=int(prob_no_damage_fr[n_haz]))
    
    Replace_time = np.random.lognormal(mean=np.log(445), sigma=0.57, size=int(prob_irrep[n_haz])) + np.full(int(prob_irrep[n_haz]), 14*n_test)
    
    DT_sip_predicted = list(Insp_time_sip)+list(Replace_time)+list(IF_SiP_sampled+RT_SiP_sampled)
    DT_fr_predicted = list(Insp_time_fr)+list(Replace_time)+list(IF_FR_sampled+RT_FR_sampled)
    DT_sip_stats_predicted = [np.mean(DT_sip_predicted), stats.variation(DT_sip_predicted), np.median(DT_sip_predicted)]
    DT_fr_stats_predicted = [np.mean(DT_fr_predicted), stats.variation(DT_fr_predicted), np.median(DT_fr_predicted)]
    
    DT_sip_actual = pd.read_csv(cd+'/S'+str(n_test)+'H14SEAWBPG2/2014/'+haz+'outputs/'+'DT_stepfunc_SiP.csv',index_col=None,header=None).iloc[2:,-1].reset_index(drop=True)
    DT_fr_actual = pd.read_csv(cd+'/S'+str(n_test)+'H14SEAWBPG2/2014/'+haz+'outputs/'+'DT_stepfunc_FR.csv',index_col=None,header=None).iloc[2:,-1].reset_index(drop=True)
    DT_sip_stats_actual = [np.mean(DT_sip_actual), stats.variation(DT_sip_actual), np.median(DT_sip_actual)]
    DT_fr_stats_actual = [np.mean(DT_fr_actual), stats.variation(DT_fr_actual), np.median(DT_fr_actual)]
  
    #FR_estimates = [IF_FR_dist,RT_FR_dist]
    #SiP_estimates = [IF_SiP_dist,RT_SiP_dist]
    DT_sip_stats = [DT_sip_stats_actual, DT_sip_stats_predicted]
    DT_fr_stats = [DT_fr_stats_actual, DT_fr_stats_predicted]
    return DT_sip_stats, DT_fr_stats, [DT_sip_predicted,DT_sip_actual], [DT_fr_predicted,DT_fr_actual]



