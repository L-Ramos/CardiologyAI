# -*- coding: utf-8 -*-
"""
This file contains methods that are used with ECG data. For some functions the models are already set, some can be optimized

@author: laramos
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, AveragePooling1D 
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling1D, Flatten, LeakyReLU,ConvLSTM2D,AveragePooling2D,AveragePooling3D,BatchNormalization
from tensorflow.keras import optimizers

from tensorflow.keras import backend as K
from losses import huber_loss,binary_focal_loss
import numpy as np

def LSTM_model_1():
    """
    This is a fixed model that worked well for PLN, but it is 2D, so one kernel is shared among all the leads
    Make sure to use 1D kernels
    Returns a trained keras model
    """

    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(1,36), activation='relu', input_shape=(4,8,256,1), name="inp/conv1",return_sequences=True))
    model.add(Dropout(0.3))
    model.add(ConvLSTM2D(filters=32, kernel_size=(1,36), activation='relu', name="conv2",return_sequences=True))
    model.add(Dropout(0.3))
    model.add(ConvLSTM2D(filters=32, kernel_size=(1,36), activation='relu', name="conv3",return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='sigmoid'))
    
    print(model.summary())
    #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    rms = optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

    return(model)




def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy
   
def Build_Model(params,input_shape,lr,w_0,w_1):
    """
    This is a function that allows you to run multiple  models.
    You can use it to define 1D/2D cnns and LSTM
    Inputs:
        params: dict
            Dictionary with all the parameters for the model, see create_json for more information
        input_shape: array
            Array with the shape of the input
        lr: int
            learning rate for the model. This is not included in the dictionary because it was controled by an outside for loop
        w_0,w_1 = float
            weights if training with imbalanced data (better to add to the compile part as class weights)            
    
    """
    

    model = Sequential()
    
    k=0
    
    if params['conv_type'] == 'LSTM':        
            model.add(ConvLSTM2D(filters=params['num_filters'][k], kernel_size=params['kernel_size'][k], activation=params['activation'], input_shape=input_shape, name='inp/conv1',return_sequences=True))            
    else:
        if params['conv_type'] == 'CNN_1D':
            model.add(Conv1D(filters=params['num_filters'][k], kernel_size=(params['kernel_size'][k][0]), activation=params['activation'], input_shape=input_shape, name='inp/conv1'))
        else:
            model.add(Conv2D(filters=params['num_filters'][k], kernel_size=params['kernel_size'][k], activation=params['activation'], input_shape=input_shape, name='inp/conv1'))
    if params['dropout'][k]!=0:
            model.add(Dropout(params['dropout'][k]))

    if params['conv_type'] == 'LSTM':                    
        if params['pool_size'][k]!=[0]:
                model.add(AveragePooling3D(pool_size=params['pool_size'][k])) 
    else:
        if params['pool_size'][k]!=[0] and params['conv_type'] == 'CNN':
                model.add(AveragePooling2D(pool_size=params['pool_size'][k])) 
        else:
            if params['pool_size'][k]!=[0] and params['conv_type'] == 'CNN_1D':
                model.add(AveragePooling1D(pool_size=params['pool_size'][k])) 
    
    for k in range(1,params['num_conv_layers']):
        name = "conv_"+str(k)
        if params['conv_type'] == 'LSTM':
            model.add(ConvLSTM2D(filters=params['num_filters'][k], kernel_size=params['kernel_size'][k], activation=params['activation'], name=name,return_sequences=True))            
        else:
            if params['conv_type'] == 'CNN_1D':
                 model.add(Conv1D(filters=params['num_filters'][k], kernel_size=(params['kernel_size'][k][0]), activation=params['activation'], name=name))    
            else:            
                model.add(Conv2D(filters=params['num_filters'][k], kernel_size=params['kernel_size'][k], activation=params['activation'], name=name))
        if params['dropout'][k]!=0:
            model.add(Dropout(params['dropout'][k]))
        if params['conv_type'] == 'LSTM':                    
            if params['pool_size'][k]!=[0]:
                    model.add(AveragePooling3D(pool_size=params['pool_size'][k])) 
        else:
            if params['pool_size'][k]!=[0]and params['conv_type'] == 'CNN':
                    model.add(AveragePooling2D(pool_size=params['pool_size'][k]))       
            else:
                if params['pool_size'][k]!=[0]and params['conv_type'] == 'CNN_1D':
                    model.add(AveragePooling1D(pool_size=params['pool_size'][k]))  
    model.add(Flatten())
    for k in range(0,params['num_dense_layers']):   
        model.add(Dense(params['size_dense_layer'][k], activation=params['activation']))
    model.add(Dense(params['num_classes'],activation = params['activation_last']))
        
                
    print(model.summary())
    #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    rms = optimizers.RMSprop(lr=lr, decay=1e-6)
    #rms = optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy',f1_m])
#    model.compile(loss=create_weighted_binary_crossentropy(w_0,w_1), optimizer=rms, metrics=['accuracy',f1_m])
    
    #model.compile(loss=binary_focal_loss(alpha=.25, gamma=2), optimizer=rms, metrics=['accuracy'])
    #model.compile(loss=huber_loss, optimizer=rms, metrics=['accuracy'])

    return(model)
    
    
    