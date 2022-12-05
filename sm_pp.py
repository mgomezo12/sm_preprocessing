import numpy as np
import h5py
import matplotlib.pyplot as plt
import datetime as dt
import scipy.io as sio
from functools import reduce
import pandas as pd


#%% 

#####        FUNCTIONS    ####################

def ymstrtodate(dfile):
    month=[f'{i[1]:02}' for i in dfile]
    year=[str(i[0]) for i in dfile]
    dateslist = [month, year]
    dates = np.apply_along_axis('-'.join, 0, dateslist)
    dateso = [dt.datetime.strptime(da, '%m-%Y').date() for da in dates]
    
    return dateso
    
#%%

#Folder of input data
pathinput="./data/"

#Read Soil moisture data observations  / stations
filenamesm='ismn.mat'
sobsf = sio.loadmat(pathinput+filenamesm)
sobs=sobsf['data_ismn_sm_monthly']
sdates=sobsf['data_ismn_t_monthly']
sid=sobsf['ismn_Lat_Lon_id_KarstLandscape']

fildepth='ismn_depth.mat'
sodepth = sio.loadmat(pathinput+fildepth)
sodeptop=sodepth['ismn_depth_sta']
sodepbot=sodepth['ismn_depth_end']

#%%
#Procedure to choose the depth
c=1
n=10
stv, depstat,datesstat =[], [], []
fig, ax =plt.subplots(10,1, figsize=(13,10), sharex=True) 
#for st in range(len(sobs)):
for st in range(10,20):
    stat=sobs[st,:]
    dstat=sdates[st,:] # read the dates
    scid= sid[st,:]  # read calibration id
    
    #Probe depths
    stop=sodeptop[st,:]
    sbot=sodepbot[st,:]
    
    
    #for p in range(len(stat)):#soil moisture prompt
    vmind, vmaxd= [],[]

    for p in range(len(stat)):
        pstat=np.hstack(stat[p])
        if pstat.size == 0:
            continue  
        dpstat=ymstrtodate(dstat[p])
        
        depthprob=sbot[p]
        line3=None
        if depthprob <0.5:
            color='#22876E'
            label='<0.5'
            line1,=ax[st-n].plot(dpstat,pstat,lw=1,color=color, label=label)
        elif depthprob <1:
            color='#D4505F'
            label='0.5-1.0'
            line2,=ax[st-n].plot(dpstat,pstat,lw=1,alpha=0.8,color=color, label=label)
        else:
            color='#7F64E3'
            label='>1.0'
            line3,=ax[st-n].plot(dpstat,pstat,lw=1,alpha=0.5,color=color, label=label)
        
        if line3:
            ax[st-n].legend([line1,line2,line3],['<0.5m','0.5-1.0m','>1.0m'], 
                          loc='upper right', fontsize='x-small')
        else:
            ax[st-n].legend([line1,line2],['<0.5m','0.5-1.0m'],
                          loc='upper right', fontsize='x-small')
        ax[st-n].grid(linewidth=.5, alpha=0.5)
        
    fig.text(0.5, .07,"Dates", ha='center')  
    
    fig.text(0.08, 0.5, 'Soil moisture observations', va='center', rotation='vertical') 


#%%      
  
c=1
n=0
stv, depstat, depstat05,datesstat, datesstat05 =[], [], [] ,[] ,[]
dfsall=[]
#fig, ax =plt.subplots(10,1, figsize=(13,10), sharex=True) 
for st in range(len(sobs)):
#for st in range(2):
    stat=sobs[st,:]
    dstat=sdates[st,:] # read the dates
    scid= sid[st,:]  # read calibration id
    
    #Probe depths
    stop=sodeptop[st,:]
    sbot=sodepbot[st,:]
    
    
    ssobs[st]
    dssobs[st]
    
    dfs, dfs05=[], []
    for p in range(len(stat)):
        pstat=np.hstack(stat[p])
        if pstat.size == 0:
            continue  
        dpstat=ymstrtodate(dstat[p])
        
        depthprob=sbot[p]
        d = {'pstat_'+str(p)+'_'+str(depthprob)[2:]: pstat}
        dfs.append( pd.DataFrame(data=d,  index= dpstat) )
        
        if depthprob <0.5:
            dfs05.append(pd.DataFrame(data=d,  index= dpstat) )
        elif not dfs05 and depthprob <1:
            dfs05.append(pd.DataFrame(data=d,  index= dpstat) )
        else:
            dfs05.append(pd.DataFrame(data=d,  index= dpstat) )
         
        

        #ax[st-n].plot(dpstat, pstat, alpha=0.8, color='lightblue')
        
    dfall=reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)  
    dfall05=reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs05) 
        
    dfall['mean']= dfall.mean(axis=1) 
    dfall05['mean']= dfall05.mean(axis=1)
    
    dsel = {'sel_pstat': ssobs[st]}
    dfall=dfall.join(pd.DataFrame(data=dsel,  index= dssobs[st]))
    
    dfsall.append(dfall)
    depstat.append(dfall['mean'].values)
    depstat05.append(dfall05['mean'].values)
    datesstat.append(np.array(dfall.index))
    datesstat05.append(np.array(dfall05.index))
    
    
    # ax[st-n].plot(dfall.index, dfall['mean'], color='darkblue', label='Mean') 
    # ax[st-n].plot(dfall05.index, dfall05['mean'], color='#D4505F', label='Mean <0.5m') 
    # ax[st-n].plot(dfall.index, dfall['sel_pstat'], color='#E0A458', label='Selected')
    # ax[st-n].grid(linewidth=.5, alpha=0.5)
    # ax[0].legend(loc='upper right', fontsize='x-small')
        
    # fig.text(0.5, .07,"Dates", ha='center')  
    # fig.text(0.08, 0.5, 'Soil moisture observations', va='center', rotation='vertical') 
    

#%%
#Save 
path= r"J:\NUTZER\GomezOspina.M\AH\input_data\code_fluxnetv2/"
np.save(path+'dfsall', np.array(dfsall,dtype=object))

#%%


# fig, ax =plt.subplots(2,2, figsize=(15,9), gridspec_kw={'width_ratios': [3, 1]}) 
# c=1
# n=0
# for st in range(2):
#     stat=sobs[st,:]
#     dstat=sdates[st,:] # read the dates
     
#     dpstatini=ymstrtodate(dstat[0])
#     pstatini=np.hstack(stat[0])
#     corrv=[]
    
#     for p in range(len(stat)):#soil moisture prompt
#    # for p in range(3):
#         pstat=np.hstack(stat[p])
#         if pstat.size == 0:
#             break
#         dpstat=ymstrtodate(dstat[p])
        
#         di=list(set.intersection(*map(set,[dpstatini,dpstat])))
#         di.sort()
#         print(len(di))
#         if len(di)==0:
#             break
#         pstati=pstat[np.in1d(dpstat,di)]
#         pstatinii=pstatini[np.in1d(dpstatini,di)]
        
#         corrv.append(np.corrcoef(pstati,pstatinii )[0][1])
        
#         ax[st-n,0].plot(dpstatini,pstatini, 'k',lw=1.5)
#         ax[st-n,0].plot(di,pstati)
#     ax[st-n,0].grid(alpha=0.2)
#     ax[st-n,0].set_ylabel('Soil moisture')
    
#     sns.histplot(data=corrv,kde=True,binwidth=0.1, ax=ax[st-n,1])
#     ax[st-n,1].set_ylabel(' ')
#     ax[st-n,1].set_xlim([0,1])
    

# plt.subplots_adjust(hspace=0.3,top=0.97, bottom=0.05, left=0.05)  


