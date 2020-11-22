# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:55:10 2020

@author: TES Installatie


For creating new datasets out of the data folder, i.e. by leaving out certain conditions such as STILL
"""

import librosa
import sys
import os
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
import crepe
from scipy.io import wavfile

sys.path.append('..') #go to current folder
path = os.path.join('..', 'Thesis2020','dataset', 'piano') #directory of dataset
label = []
# mfccs = []
# mfccDeltas = []
pitch = []
chromagrams = []
tempograms = []
onsets = []
rms = []
mocaps = []
features = []
maxLens = []
wsize =  np.round(1000/100) #1000/framerate mocap
# for counting number of frames per condition
FE = 0
FL = 0
FN = 0
FS = 0
NE = 0
NL = 0
NN = 0
NS = 0
SE = 0
SL = 0
SN = 0
SS = 0
RE = 0
RL = 0
RN = 0
RS = 0

for subdir, dirs, files in os.walk(path): #go through all subfolders
 # Leave out still category, only look at pianist 1
  if not "_STILL" in subdir:
   if not "pianist02" in subdir:
    for file in files:
     p = os.path.join(subdir, file) #go through files
     if "mocap" in p:
        data = pd.read_csv(p, sep=';')
        frames = np.size(data,0)
		#count frames to take into account for i.e. training
        if "FAST" in p:
            if "EXAG" in subdir:
                 FE+=frames
            if "LEG" in subdir:
                FL+=frames
            if "STAC" in subdir:
                FS+=frames
            if "NORMAL" in subdir:
                FN+=frames
        if "NORMAL_EXAG" in subdir:
            NE+=frames
        if "NORMAL_LEG" in subdir:
            NL+=frames
        if "NORMAL_STAC" in subdir:
            NS+=frames
        if "NORMAL_NORMAL" in subdir:
            NN+=frames
        if "SLOW" in subdir:
            if "EXAG" in subdir:
                SE+=frames
            if "LEG" in subdir:
                SL+=frames
            if "STAC" in subdir:
                SS+=frames
            if "NORMAL" in subdir:
                SN+=frames
        if "RUBATO" in subdir:
            if "EXAG" in subdir:
                RE+=frames
            if "LEG" in subdir:
                RL+=frames
            if "STAC" in subdir:
                RS+=frames
            if "NORMAL" in subdir:
                RN+=frames
                # for file in files:
                 
                    # p = os.path.join(subdir, file) #go through files                                     
                     if "audio.wav" in p: # extract music feature
                         print(p)
                         music, Fs = librosa.load(p) #load audio for feature extraction
               
                     if "melodia" in p:
                         m = pd.read_csv(p, sep=',')
                         melody = m.iloc[:,1].values
                       
                     if "mocap" in p:
                         feature = np.array([])
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
                         mocaps.append(mocap)           
                         S = np.size(data,0) # number of frames in mocap data
                         M = np.size(music)  # number of samples in audio data
                         h = np.int(np.round(wsize - ((S+1)*wsize - M)/S)) #find ideal hop length so audio feature is aligned to framerate of mocap data
                         mfcc = librosa.feature.mfcc(music, sr=Fs, n_mfcc=16, hop_length=h)                    
                         pitch = np.interp(np.arange(0, len(melody), 3.439), np.arange(0, len(melody)), melody)
                   
                         maxLen = (min(len(pitch), len(mfcc[1,:])))
                         maxLens.append(maxLen)
                         feature = np.concatenate((mfcc[:, 0:maxLen], librosa.feature.delta(mfcc[:, 0:maxLen])))
                         feature = np.concatenate((feature, [pitch[0:maxLen]]))
                         onsetStrength = librosa.onset.onset_strength(music, sr=Fs, hop_length=h) #find onsets
                         onsetInd = librosa.onset.onset_detect(music, sr=Fs, hop_length=h, onset_envelope=onsetStrength) #find beat onsets
                         onsetFrames = np.zeros(np.size(mfcc,1))
                         onsetFrames[onsetInd] = 1   #Mark frames with an onset with 1
                         feature= np.concatenate((feature,[onsetFrames[0:maxLen]]))        
                         onsets.append(onsetFrames)        
                         rms = librosa.feature.rms(music, hop_length=h)[0]
                         feature = np.concatenate((feature, [rms[0:maxLen]]))    #compute root mean square
                         label.append(''.join(subdir[46:])) #get condition name
                         features.append(feature)
 for i in range(0,np.size(mocaps)):
     if(np.size(mocaps[i],0) > maxLens[i]):
         mocaps[i] = mocaps[i][0:maxLens[i],:] # clip mocap to match audio feature length, mocap is sometimes longer (pianist stays still while sound already stopped)
     if(np.size(mocaps[i],0) < maxLens[i]):
         last = mocaps[i][-1,:]
         diff = maxLens[i] -np.size(mocaps[i],0)
         for p in range(0,diff):        
             mocaps[i] = np.concatenate((mocaps[i],[last])) # extend last frame of mocap to match audio feature length (sometimes mocap is cut earlier on the standing still frame while sound is still fading)
     print("mocapsize:" + str(np.size(mocaps[i],0)) )
     print(maxLen)
aa= [FE,FL,FN,FS,NE,NL,NN,NS,SE,SL,SN,SS,RE,RL,RN,RS]
for a in aa:
    print(a)


label = np.array(label) #add condition name as label, i.e. fast-normal
dataset = pd.DataFrame(np.transpose([mocaps, features]), columns=['MoCap', 'Feature vector'], index=label)
dataset.to_pickle(os.path.join('..','Thesis2020','dataFeaturesP1_ALL')) #name new created dataset