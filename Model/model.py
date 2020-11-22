# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:30:40 2020

@author: TES Installatie
"""
from sklearn.decomposition import PCA
import pandas as pd
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Input, optimizers, activations, regularizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, TimeDistributed, Activation, Bidirectional, Conv3D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, History 
from sklearn.preprocessing import MinMaxScaler, normalize
from kerastuner.tuners import RandomSearch
from keras.backend import sigmoid
import keras.backend as K
from matplotlib import pyplot
from keras.utils.generic_utils import get_custom_objects
import sklearn
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
sys.path.append('..') #go to current folder
data = pd.read_pickle(os.path.join('..','Thesis2020','dataFeaturesP1_stacleg')) # Load data in format [MoCap, Features: MFCC[0:16] MFCC-Delta[16:32] Pitch[32] Beat-Onsets[33] RMS[34]]
shuffled = data.sample(frac=1,random_state=13)  #shuffle data so the conditions aren't in order anymore (for random samples)
dataset = shuffled.values # convert to arrays
features = dataset[:,1]
featureScaler = MinMaxScaler((-1,1))
scalery = MinMaxScaler((-1,1))
velocityScaler = MinMaxScaler((-1,1))
X_mfcc = []
y_mfcc = []
norm = []
y = dataset[:,0]
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
def getVelocity_PastFrame(a, b):
    velocities = np.zeros((len(a),42))
    i=0
    while i <42:
        velocities[1:,i] = (np.diff(a[:,i]))
        velocities[1:,i+1]  = (np.diff(a[:,i+1]))
        velocities[1:,i+2]  = (np.diff(a[:,i+2]))
        i+=3
    vels = np.concatenate((b,velocities), axis=1)
    return vels


for i in range(0,len(dataset)):
	#if using multiple adjacent features
    feature = features[i][32:].T
	
	# # if using a single column feature
	# feature = features[i][32]
    # feature = feature.reshape(-1, 1)
	
	# # if using 2 non-adjacent features
    # feature = np.vstack((features[i][32],features[i][34])).T 
  
    mc = y[i]
    X_mfcc.append(feature) 
    y_mfcc.append(mc); 
    

# scale features so they are within the range of -1 to 1 (easier to train for the network), these scalers can be also be used to scale back generated results
__ = [featureScaler.partial_fit(x) for x in X_mfcc]
X_scaled = [featureScaler.transform(x) for x in X_mfcc]  
__ = [scalery.partial_fit(y) for y in y_mfcc]
y_scaled = [scalery.transform(y) for y in y_mfcc]  


testInd  = [0,1,2,3,4,5,8,9,10,11,12,13,20,21,23,56] # hand-picked test set to make sure it contains one of each test category
testset =   np.zeros(np.size(y_scaled))
testset[testInd] = 1
X_train = []
X_test =[]
y_train = []
y_test = []
indd = 0
for things in testset:
    if things == 1:
        X_test.append(X_scaled[indd])
        y_test.append(y_scaled[indd])
    else:
        X_train.append(X_scaled[indd])
        y_train.append(y_scaled[indd])
    indd+=1

# Parameters for network
dimension = np.size(X_train[0],1)
max_size = 3200
pad = -10.0
step_size = 300

# split train and validation set
val_split = int(len(X_train)*0.85)
X_train, X_val = X_train[:val_split], X_train[val_split:]
y_train, y_val = y_train[:val_split], y_train[val_split:]


def makeSequenceY(a):
    seq = []
    firstFrame = 0
    while not firstFrame > int(len(a)/step_size) * step_size-step_size:
        frame = a[firstFrame:firstFrame+step_size]
        seq.append(frame)
        firstFrame = firstFrame+step_size
    seqr = np.asarray(seq)
    print(len(seqr))
    return seqr.reshape((len(seq), step_size, np.size(a,1)))

def makeSequenceX(a):
    seq = []
    firstFrame = 0    
    while not firstFrame > int(len(a)/step_size) * step_size-step_size:
        temp = a[firstFrame:firstFrame+step_size]
        seq.append([temp])
        firstFrame = firstFrame+step_size
    seqr = np.asarray(seq)
    print(len(seqr))
    return seqr.reshape((len(seq), step_size, dimension))

# cut the training data into pieces based on the step_size
def makeSequenceSets():
    seq_X_train = []
    for x in X_train:
        seq_X_train.append(makeSequenceX(x))
    seq_X_train = np.asarray(seq_X_train)
    
    seq_y_train = []
    for y in y_train:
        seq_y_train.append(makeSequenceY(y))
    seq_y_train = np.asarray(seq_y_train)      
    
    seq_X_val = []
    for x in X_val:
        seq_X_val.append(makeSequenceX(x))
    seq_X_val = np.asarray(seq_X_val)
       
    seq_y_val = []
    for y in y_val:
        seq_y_val.append(makeSequenceY(y))
    seq_y_val = np.asarray(seq_y_val)
        
    seq_X_test = []
    for x in X_test:
        seq_X_test.append(makeSequenceX(x))
    seq_X_test = np.asarray(seq_X_test)  
       
    seq_y_test = []
    for y in y_test:
        seq_y_test.append(makeSequenceY(y))     
    seq_y_test = np.asarray(seq_y_test)          
        
    return seq_X_train, seq_y_train, seq_X_test, seq_y_test, seq_X_val, seq_y_val#


seq_X_train, seq_y_train, seq_X_test, seq_y_test, seq_X_val, seq_y_val = makeSequenceSets()#

history = History()

# Set learning rate manually, gets overwritten when using cyclical learning rates below
optLr = 0.0001

# # Run a dummy network to find the optimal learning rate using cyclical learning rates
# motionModel = Sequential([LSTM(input_shape=(step_size,42+42+dimension),units=32, return_sequences=True),TimeDistributed(Dense(42, activation='tanh'))])
# print(motionModel.summary())
# start_lr = 0.0001
# end_lr = 0.01
# num_batches = len(seq_X_train)
# veloc = np.zeros((step_size,42))
# cur_lr = start_lr
# lr_multiplier = (end_lr / start_lr) ** (1.0 / num_batches)
# lossess = []
# losses = []
# lrs = []
# print('finding optimal learning rate')
# for sampleNr in range(num_batches):
#     print(sampleNr)
    
#     motionModel.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=cur_lr))
#     sampleX = seq_X_train[sampleNr]
#     sampley = seq_y_train[sampleNr]
#     for frame in range(len(sampleX)):
#               xTrain = np.array([sampleX[frame]])
#               yTrue = np.array([sampley[frame]])
#               if frame == 0:
#                   veloc =  np.zeros((step_size,42))
#                   concat = np.array([np.concatenate((np.concatenate((sampleX[frame],yTrue[0]),axis=1),veloc),axis=1)])
#                   motionModel.fit(concat,yTrue, epochs=1,verbose=0)
#                   newMotion = motionModel.predict(concat)
#                   veloc = yTrue[0] - newMotion[0]
#                   loss = motionModel.history.history['loss'][0]
#                   losses.append(loss)

#               else:
#                   oldFrame = newMotion[0]
#                   concat = np.array([np.concatenate((np.concatenate((sampleX[frame],newMotion[0]),axis=1),veloc),axis=1)])
#                   history = motionModel.fit(concat,yTrue, epochs=1,verbose=0, callbacks=([history]))
#                   newMotion = motionModel.predict(concat)
#                   veloc = oldFrame-newMotion[0]
#                   loss = motionModel.history.history['loss'][0]
#                   losses.append(loss)
            
#     lossess.append(np.mean(losses))
#     lrs.append(cur_lr)
#     cur_lr = cur_lr*lr_multiplier # increase LR

# max_slope = [x - z for x, z in zip(lrs, lossess)]    
# optLr = lrs[np.where(min(max_slope)==max_slope)[0][0] ]
# print('lr:', optLr)


# Scale back the generated and ground truth results to see absolute distance for error measurements
def inv(truth, predicted):
    truth = scalery.inverse_transform(truth)
    predicted = scalery.inverse_transform(predicted)
    return truth, predicted

# Network creation
motionModel = Sequential([LSTM(input_shape=(step_size,42+42+dimension),units=64, return_sequences=True),LSTM(units=64, return_sequences=True),TimeDistributed(Dense(42, activation='tanh'))])
motionModel.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=optLr))
trainlosses = []
vallosses = []
meanLosstrain = []
meanLossval = []
epochs = 50
penalty=0
for i in range(epochs):
    print('epoch',i)
	
    print('training')
    for sampleNr in range(len(seq_y_train)):     
        sampleX = seq_X_train[sampleNr]
        sampley = seq_y_train[sampleNr]
        prediction = []
        valPred = []
        prediction.append(sampley[0][0])
        for frame in range(len(sampleX)):
            xTrain = np.array([sampleX[frame]])
            yTrue = np.array([sampley[frame]])
            if frame == 0: # train on ground truth for frame 0
                veloc =  np.zeros((step_size,42))
                concat = np.array([np.concatenate((np.concatenate((sampleX[frame],yTrue[0]),axis=1),veloc),axis=1)])
                motionModel.fit(concat,yTrue, epochs=1,verbose=0)
                newMotion = motionModel.predict(concat) # set generated result as new old frame
                veloc = yTrue[0] - newMotion[0] # compute velocity between old and new frame, i.e. how drastic transition is (this is one of the training features)
                prediction = newMotion[0] 

            else: # train on generated previous frame + velocity to current frame + features
                oldFrame = newMotion[0]
                concat = np.array([np.concatenate((np.concatenate((sampleX[frame],newMotion[0]),axis=1),veloc),axis=1)])
                history = motionModel.fit(concat,yTrue, epochs=1,verbose=0, callbacks=([history]))
                newMotion = motionModel.predict(concat)
                veloc = oldFrame-newMotion[0]
                prediction= np.concatenate((prediction,newMotion[0]))
        scaleY, scalePred = inv(y_train[sampleNr][:len(prediction)], prediction)
        mse = sklearn.metrics.mean_squared_error(scaleY[:len(prediction)], scalePred)
        trainlosses.append(mse)
        
    print('validating')            
    for sampleNr in range(len(seq_y_val)):  
        sampleXval = seq_X_val[sampleNr]
        sampleyval = seq_y_val[sampleNr]
        for frame in range(len(sampleXval)):
             xVal = np.array([sampleXval[frame]])
             yVal = np.array([sampleyval[frame]])
             if frame == 0:
                veloc =  np.zeros((step_size,42))
                concat = np.array([np.concatenate((np.concatenate((sampleXval[frame],yVal[0]),axis=1),veloc),axis=1)])
                loss = motionModel.evaluate(concat,yVal, verbose=0)
                newMotion = motionModel.predict(concat)
                veloc = yVal[0] - newMotion[0]
                valPred= newMotion[0]

             else:
                oldFrame = newMotion[0]
                concat = np.array([np.concatenate((np.concatenate((sampleXval[frame],newMotion[0]),axis=1),veloc),axis=1)])
                loss = motionModel.evaluate(concat,yVal, verbose=0, callbacks=([history]))
                newMotion = motionModel.predict(concat)
                veloc = oldFrame-newMotion[0]
                valPred= np.concatenate((valPred,newMotion[0]))
        
        scaleYval, scalePredval = inv(y_val[sampleNr][:len(valPred)], valPred)  
        msev = sklearn.metrics.mean_squared_error(scaleYval[:len(valPred)], scalePredval)
        trainlosses.append(mse)
        vallosses.append(msev)
    meanLosstrain.append(np.mean(trainlosses))
    meanLossval.append(np.mean(vallosses))
    trainlosses = []
    vallosses = []
    print('loss:' , meanLosstrain[-1])
    print('val loss:' , meanLossval[-1])
    if i >2:
        if (meanLossval[-1] > meanLossval[-2]):
            penalty+=1
            if   (meanLossval[-1] - meanLossval[-2] > 50):
                penalty+=1
    if penalty >= 5:
        break
    print('penalty = ', penalty)

# Plot training and validation errors against each other (Validation should be better than training but generally they should not differ too much or something is wrong with the network, i.e. overfitting/underfitting/bad data)
pyplot.figure
pyplot.plot(meanLosstrain)
pyplot.plot(meanLossval)
pyplot.title('Pitch+Beat+RMS')
pyplot.legend(['Train', 'Validation'])
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.show()

# Get the corresponding labels for all the test data
label = 0
labels = []
for labell in testInd:
    labels.append(shuffled.index[labell])

# Smoothing filter for results, N is over how many frames you want to smooth out
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
	
# Run network on test data and save results	
for i in range(len(seq_X_test)):
    output = []
    x = seq_X_test[i]
    y = seq_y_test[i][0]
    for frame in range(len(x)):
        test = np.array([x[frame]])
        if frame == 0 :        
            veloc = np.zeros((step_size,42))
            new = motionModel.predict(np.array([np.concatenate((np.concatenate((x[frame],y),axis=1),veloc),axis=1)]))
            output = new[0]
            oldFrame = new[0]

        else:
            
            test = np.array([np.concatenate((np.concatenate((x[frame],oldFrame),axis=1),veloc),axis=1)])
            new = motionModel.predict(test)
            output = np.concatenate((output,new[0]))
            veloc = oldFrame-new[0]
            oldFrame= new[0]

    output = scalery.inverse_transform(output)
    c=np.zeros(np.shape(output))
    for i in range(42):
            c[:,i] = runningMeanFast(output[:,i], 50).T
    np.savetxt(os.path.join("results",'pianist1', 'stacc_leg_norm_exag','pitchbeatrms', labels[label]+'.csv'), c, delimiter=';') # save results as csv-file, folder names go in here
    label+=1

# # Shut down PC when done with training
# os.system("shutdown /s /t 1") 
