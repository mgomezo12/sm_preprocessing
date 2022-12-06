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

def plateu(df, p, depth,wd=3, th=0.01):
    dfv, dfd=[], []
    for i in range(len(df)):
        if i <= wd:
            sec=df.values[i:i+wd*2]
        elif i>=len(df)-wd:
            sec=df.values[i-wd*2:i]
        else:
            sec=df.values[i-wd:i+wd]
        y1=abs(sec[0]-sec[wd])
        y2=abs(sec[wd]-sec[-1])
        y3=abs(sec[-1]-sec[0])
        dym=np.mean([y1,y2,y3])
        if dym> th:
            dfv.append(df.values[i][0])
            dfd.append(df.index[i])
            #print(dym)
        
        d = {'pstat_'+str(p)+'_'+str(int(depth*100)): dfv}
        dff=pd.DataFrame(data=d,  index= dfd) 
        
    return dff

def stdout(df):
    df1=df[df<df.mean()+3*df.std()]
    df2=df1[df>df.mean()-3*df.std()]
    df3=df2.dropna()
    return df3    
    
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
 
c=1
n=0
stv, depstat, depstat05,datesstat, datesstat05 =[], [], [] ,[] ,[]
dfsall=[]
#fig, ax =plt.subplots(10,1, figsize=(13,10), sharex=True) 
#for st in range(len(sobs)):
count=0
for st in range(20):
    stat=sobs[st,:]
    dstat=sdates[st,:] # read the dates
    scid= sid[st,:]  # read calibration id
    
    #Probe depths
    stop=sodeptop[st,:]
    sbot=sodepbot[st,:]

    
    dfs, dfs05=[], []
    for p in range(len(stat)):
        pstat=np.hstack(stat[p])
        if pstat.size == 0:
            continue  
        dpstat=ymstrtodate(dstat[p])
        
        depthprob=sbot[p]
        d = {'pstat_'+str(p)+'_'+str(int(depthprob*100)): pstat}
        df1=pd.DataFrame(data=d,  index= dpstat) 
        df2=df1.dropna()
        df3=df2[(df2<1)&(df2>0)]
        
        df4=plateu(df3, p=p, depth= depthprob, wd=3,th=abs(df3.diff()).mean()[0])
        
        df5=stdout(df4)
        
        dfs.append(df5)

        if depthprob <0.5:
            dfs05.append(df5)
        elif not dfs05 and depthprob <1:
            dfs05.append(df5)
        else:
            dfs05.append(df5)
        
    dfall=reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)  
    dfall05=reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs05) 
        
    dfall['mean']= dfall.mean(axis=1) 
    dfall05['mean']= dfall05.mean(axis=1)
    
    dfall05['meanf']=stdout(dfall05['mean'])
    
    if len(dfall05['mean'].dropna())<18:
        count+=1
         
    df6=stdout(df5)
    
    plt.figure()
    plt.plot(dfall05['mean'],'.-', color="chocolate", lw=0.4)
    plt.plot(dfall05['meanf'],'.-', color="blue", lw=0.4)
    
    
    
    # dfsall.append(dfall)
    # depstat.append(dfall['mean'].values)
    # depstat05.append(dfall05['mean'].values)
    # datesstat.append(np.array(dfall.index))
    # datesstat05.append(np.array(dfall05.index))

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



