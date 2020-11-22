# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:32:12 2020

@author: TES Installatie
"""
import pandas as pd
import sys
import os
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import scipy 
sys.path.append('..') #go to current folder
folderName = 'no_rubatos" #name of folder to analyse
path = os.path.join('..', 'Thesis2020','results', 'pianist1', folderName') #directory of dataset
label = []
allF = []
ground = []
mfccs = []
mfccRMSBeat = []
pitchRMSBeat = []
pitchRMS = []
pitchBeat = []
pitches = []
for dirs in os.scandir(path):
    for root, subs, files in os.walk(dirs.path):

             if 'all' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(root, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       allF.append(np.asarray(data))
             if 'ground' in root:
               for file in files:
                   if '.csv' in file:
                        p = os.path.join(root, file) #go through files  
                        data = pd.read_csv(p, sep=';')
                        hip_pos = np.array(data.iloc[:,0:3]) # use hip position to normalize data (put hip to (0,0,0) and move everything else relative to this)
                        markers = np.array([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]) # bones used to animate the figure
                        mocap = np.zeros((np.size(data,0),42))
                        for m in markers: 
                            i = np.int(m/7*3)
                            newPos = np.array(data.iloc[:,m:m+3]) - hip_pos #move bone xyz locations relative to new hip bone position
                            mocap[:,i] = newPos[:,0]
                            mocap[:,i+1] = newPos[:,1]
                            mocap[:,i+2] = newPos[:,2]
                        ground.append(mocap)
             if 'onlyPitch' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(dirs.path, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       pitches.append(np.asarray(data))
             if 'pitchRMSbeat' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(dirs.path, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       pitchRMSBeat.append(np.asarray(data))
             if 'withRMS' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(dirs.path, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       pitchRMS.append(np.asarray(data))
             if 'withBeat' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(dirs.path, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       pitchBeat.append(np.asarray(data))
             if 'mfcc' in dirs.path:
               for file in files:
                   if '.csv' in file:
                       p = os.path.join(root, file) #go through files  
                       data = pd.read_csv(p, sep=';')               
                       mfccs.append(np.asarray(data))

     
# uses same smoothing as used on generated results, for comparison purposes (ground truth sometimes is very jittery)
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]     
label=0
smoothgt=[]
for gt in ground:
    output = gt
    c=np.zeros(np.shape(output))
    for i in range(42):
            c[:,i] = runningMeanFast(output[:,i], 50).T
    # np.savetxt(os.path.join("results",'pianist1', folderName,'ground truth', 'smooth_'+ names[label]+'.csv'), c, delimiter=';') #uncomment to save the smoothed results for later use
    smoothgt.append(c)
    label+=1



def getQoM(mc, joint): # Quantity of motion, unused metric
    QoMs = []
    for x in mc.T[joint:joint+3]:
        QoMs.append(round(sum(abs(x)),3))        
    return QoMs
   
   
# Important columns are: head = 15, neck = 12, abdomen = 3, chest = 6, torso = 9
# Coordinates are organized like: ZXY

#### Below lines are for reorganizing the order of the results, for easy comparison. Alphabetically sorted, i.e. Fast->Rubato and Fast-Exag -> Fast->Stac.
# 0fast-exag, 1normal-exag, 2normal-normal. 3slow-exag, 4slow-normal, 5fast-normal, 6fast-normal, 7rubato-exag, 8rubato-normal, 9slow-exag

#  #NO_NORMAL
# aa = [0,4,5,6,7,1,2,8,3]
# names = ['FAST_EXAG', 'NORMAL_STAC', 'SLOW_EXAG', 'SLOW_STAC', 'FAST_LEG', 'FAST_STAC', 'NORMAL_EXAG', 'NORMAL_LEG', 'SLOW_LEG']

#  # NO_EXAG
# aa = [3,4,5,6,7,0,8,1,2]
# names = ['NORMAL_STAC', 'SLOW_NORM', 'SLOW_STAC', 'FAST_LEG', 'FAST_NORM', 'FAST_STAC', 'NORMAL_LEG', 'NORMAL_NORM', 'SLOW_LEG']

#  # ALL
# aa = [0,6,7,1,2,8,3,13,4,14,5,9,10,11,12]
# names = ['FAST_EXAG', 'FAST_STAC','NORMAL_EXAG','NORMAL_STAC','SLOW_LEG','SLOW_STAC','FAST_LEG','FAST_NORMAL','NORMAL_LEG','RUBATO_EXAG','RUBATO_LEG','RUBATO_NORMAL','RUBATO_STAC','SLOW_EXAG','SLOW_NORMAL']

# # no-rubato
aa = [0,5,6,7,1,3,4,8]
names = ['FAST_NORMAL', 'NORMAL_STAC','SLOW_EXAG','SLOW_LEG','SLOW_NORMAL','FAST_STAC','NORMAL_EXAG','NORMAL_LEG','SLOW_STAC']

    
def getEuc(x,y): # get euclidean distance for computation of average position error (APE)
    D = np.size(x,1)
    T = np.size(x,0)
    return np.linalg.norm(x-y)/(D*T)

# computes velocity per limb  per frame    
GTvels = []
for gt in smoothgt:
    GTvels.append(np.diff(gt,axis=0))
allvels = []
for a in allF:
    allvels.append(np.diff(a,axis=0))
pitchvels = []
for pitch in pitches:
    pitchvels.append(np.diff(pitch,axis=0))
pitchRMSvels = []
for pitchR in pitchRMS:
    pitchRMSvels.append(np.diff(pitchR,axis=0))
pitchBvels = []
for pitchB in pitchBeat:
    pitchBvels.append(np.diff(pitchB,axis=0))
pitchBRvels = []
for pitchBR in pitchRMSBeat:
    pitchBRvels.append(np.diff(pitchBR,axis=0))
mfccvels = []
for mfcc in mfccs:
    mfccvels.append(np.diff(mfcc,axis=0))

	
# computes acceleration per limb per frame    
GTaccs = []
for gt in GTvels:
    GTaccs.append(np.diff(gt/0.001,axis=0))
allaccs = []
for a in allvels:
    allaccs.append(np.diff(a/0.001,axis=0))
pitchaccs = []
for pitch in pitchvels:
    pitchaccs.append(np.diff(pitch/0.001,axis=0))
pitchRMSaccs = []
for pitchR in pitchRMSvels:
    pitchRMSaccs.append(np.diff(pitchR/0.001,axis=0))
pitchBaccs = []
for pitchB in pitchBvels:
    pitchBaccs.append(np.diff(pitchB/0.001,axis=0))
pitchBRaccs = []
for pitchBR in pitchBRvels:
    pitchBRaccs.append(np.diff(pitchBR/0.001,axis=0))
mfccaccs = []
for mfcc in mfccvels:
    mfccaccs.append(np.diff(mfcc/0.001,axis=0))
    
# computes jerkiness per limb per frame       
GTjerk = []
for gt in GTaccs:
    GTjerk.append(np.diff(gt/0.001,axis=0))
alljerk = []
for a in allaccs:
    alljerk.append(np.diff(a/0.001,axis=0))
pitchjerk = []
for pitch in pitchaccs:
    pitchjerk.append(np.diff(pitch/0.001,axis=0))
pitchRMSjerk = []
for pitchR in pitchRMSaccs:
    pitchRMSjerk.append(np.diff(pitchR/0.001,axis=0))
pitchBjerk = []
for pitchB in pitchBaccs:
    pitchBjerk.append(np.diff(pitchB/0.001,axis=0))
pitchBRjerk = []
for pitchBR in pitchBRaccs:
    pitchBRjerk.append(np.diff(pitchBR/0.001,axis=0))
mfccjerk = []
for mfcc in mfccaccs:
    mfccjerk.append(np.diff(mfcc/0.001,axis=0))


results = np.zeros((18,len(allF)))
GTresults= np.zeros((2, len(allF)))
print('ERROR--------------------------')
indexx = 0
for i in aa:
    print(names[i])
    print('all: ')
    print(getEuc(allF[i], smoothgt[i][:len(allF[i]),:]))
    print('mfcc: ')
    print(getEuc(mfccs[i],smoothgt[i][:len(mfccs[i]),:]))
print('ACC---------------------------')
for i in aa:
    print(names[i])
    print('all: ')
    print(np.mean(allaccs[i]))
    print('mfcc: ')
    print(np.mean(mfccaccs[i]))
    print('GT:')
    print(np.mean(GTaccs[i]))
    results[0, indexx] = np.mean(pitchaccs[i])
    results[1, indexx] = np.mean(pitchBaccs[i])
    results[2, indexx] = np.mean(pitchRMSaccs[i])
    results[3, indexx] = np.mean(pitchBRaccs[i])
    results[4, indexx] = np.mean(mfccaccs[i])
    results[5, indexx] = np.mean(allaccs[i])
    
    results[6, indexx] = np.mean(pitchjerk[i])
    results[7, indexx] = np.mean(pitchBjerk[i])
    results[8, indexx] = np.mean(pitchRMSjerk[i])
    results[9, indexx] = np.mean(pitchBRjerk[i])
    results[10, indexx] = np.mean(mfccjerk[i])
    results[11, indexx] = np.mean(alljerk[i])
    

    results[12, indexx] = getEuc(pitches[i], smoothgt[i][:len(pitches[i]),:])
    results[13, indexx] = getEuc(pitchBeat[i], smoothgt[i][:len(pitchBeat[i]),:])
    results[14, indexx] = getEuc(pitchRMS[i],smoothgt[i][:len(pitchRMS[i]),:])
    results[15, indexx] = getEuc(pitchRMSBeat[i],smoothgt[i][:len(pitchRMSBeat[i]),:])
    results[16, indexx] = getEuc(mfccs[i],smoothgt[i][:len(mfccs[i]),:])
    results[17, indexx] = getEuc(allF[i], smoothgt[i][:len(allF[i]),:])
    
    GTresults[0, indexx] = np.mean(GTaccs[i])
    GTresults[1, indexx] = np.mean(GTjerk[i])
    indexx+=1
print('JERK---------------------------')
for i in aa:
    print(names[i])
    print('all: ')
    print(np.mean(alljerk[i]))
    print('mfcc: ')
    print(np.mean(mfccjerk[i]))
    print('GT:')
    print(np.mean(GTjerk[i][:len(mfccjerk[i]),:]))