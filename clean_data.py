# -*- coding: utf-8 -*-
"""
Fixes the data in the right format for using with

@author: laramos
"""
import tensorflow.keras as keras
import numpy as np


def Clean_Data_LSTM(train_index, val_index,test_index,beat_list, target_values):
    
        X_tr, y_tr = [], []
        for idx in train_index:
            #for pat in beat_list[idx]:
                    X_tr.append(np.array(beat_list[idx]).T)
                    y_tr.append(target_values[idx])
        
        X_val, y_val = [], []
        for idx in val_index:
            #for pat in beat_list[idx]:
                    X_val.append(np.array(beat_list[idx]).T)
                    y_val.append(target_values[idx])
        
        
        X_tes, y_tes = [], []
        for idx in test_index:
            #for pat in beat_list[idx]:
                    X_tes.append(np.array(beat_list[idx]).T)
                    y_tes.append(target_values[idx])
                    
       
    
        print(np.bincount(y_tr), np.bincount(y_val), np.bincount(y_tes))
    
        #del train_index, test_index, val_index, idx
    
        X_tr, X_tes, X_val = np.array(X_tr), np.array(X_tes), np.array(X_val)
        y_tr, y_tes, y_val = np.array(y_tr), np.array(y_tes), np.array(y_val)
        X_tes = np.swapaxes(X_tes,1,3)
        X_val = np.swapaxes(X_val,1,3)
        X_tr = np.swapaxes(X_tr,1,3)
        
        X_tr = X_tr/np.max(X_tr)
        X_tes = X_tes/np.max(X_tes)
        X_val = X_val/np.max(X_val)
#        X_tr = np.swapaxes(X_tr,2,3)
#        X_tes = np.swapaxes(X_tes,2,3)
#        X_val = np.swapaxes(X_val,2,3)
        #X_tr = X_tr.reshape(-1,4,8,256)
        #X_tes = X_tes.reshape(-1,4,8,256)
        #X_val = X_val.reshape(-1,4,8,256)
    
        # Undersampling the majority
        
        y_tr = keras.utils.to_categorical(y_tr, 2)
        y_val = keras.utils.to_categorical(y_val, 2)
        y_tes = keras.utils.to_categorical(y_tes, 2)
    
        X_tr = np.expand_dims(X_tr,axis=4)
        X_tes = np.expand_dims(X_tes,axis=4)
        X_val = np.expand_dims(X_val,axis=4)
        return(X_tr,X_tes,X_val,y_tr,y_val,y_tes)
    
    
        
def Clean_Data_CNN(train_index, val_index,test_index,beat_list, target_values,cnn_1d):
    
        X_tr, y_tr = [], []
        for idx in train_index:
            for pat in beat_list[idx]:
                    X_tr.append(np.array(pat).T)
                    y_tr.append(target_values[idx])
        
        X_val, y_val = [], []
        for idx in val_index:
            for pat in beat_list[idx]:
                    X_val.append(np.array(pat).T)
                    y_val.append(target_values[idx])
                   
        X_tes, y_tes = [], []
        for idx in test_index:
            beats = beat_list[idx]
            pat = beats[int(np.round(len(beats)/2))] #getting only the beat in the middel
            X_tes.append(np.array(pat).T)
            y_tes.append(target_values[idx]) 
    
        print(np.bincount(y_tr), np.bincount(y_val), np.bincount(y_tes))
    
        del train_index, test_index, val_index, idx
    
        X_tr, X_tes, X_val = np.array(X_tr), np.array(X_tes), np.array(X_val)
        y_tr, y_tes, y_val = np.array(y_tr), np.array(y_tes), np.array(y_val)
    
        # Undersampling the majority
    #    rus = RandomUnderSampler(return_indices=True,random_state=seed)
    #    _, _, idx = rus.fit_sample(y_tr.reshape(-1, 1), y_tr)
    #    X_tr, y_tr = X_tr[idx], y_tr[idx]
    #    print(np.bincount(y_tr), np.bincount(y_tes))
        
        X_tr = X_tr/np.max(X_tr)
        X_tes = X_tes/np.max(X_tes)
        X_val = X_val/np.max(X_val)
        if not cnn_1d:
            X_tr = np.swapaxes(X_tr,1,2)
            X_tes = np.swapaxes(X_tes,1,2)
            X_val = np.swapaxes(X_val,1,2)
        #X_val = X_val.reshape(-1,8,256)                
        
        y_tr = keras.utils.to_categorical(y_tr, 2)
        y_val = keras.utils.to_categorical(y_val, 2)
        y_tes = keras.utils.to_categorical(y_tes, 2)
        if not cnn_1d:
            X_tr = np.expand_dims(X_tr,axis=3)
            X_tes = np.expand_dims(X_tes,axis=3)
            X_val = np.expand_dims(X_val,axis=3)
        #last arguemtn is the shape of the input
        return(X_tr,X_tes,X_val,y_tr,y_val,y_tes,X_tr.shape[1:])        