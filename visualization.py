# -*- coding: utf-8 -*-
"""

This computes grad cam results and occlusion maps for our models

TODO:
    This was used together with the main code, so some libraries might not be imported
    Needs second test

@author: laramos
"""
from keras.models import load_model
import pickle
from grad_cam import grad_cam_mine
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pylab as pylab
import os
import glob

import clean_data as cd
import models as md

#%%

#loads different data depending on the approach

#class to keep the evaluation measures, per fold we append a new measure
class Measures():
    def __init__(self):
        self.acc = list()
        self.spec = list()
        self.sens = list()
        self.auc_list = list()
        self.predictions = list()
        self.labels = list()
        self.auc = list()
        self.f1 = list()

### Data pre-processsing for model



init_seed = 36


model_type="CNN"    

conv_1d = True
  

data_type = "PLN"
#data_type = "PLN_imbalanced"
#data_type = "AF"

results_folder = r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\testSoftmax_Results_seed36"+data_type+"_"+model_type+"100_epochs\\"
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

if data_type == "PLN":
    data_path = r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_PLN\\"
    if model_type=="CNN": 
        if not conv_1d:
            params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\CNN_model*")
        else:
            params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\CNN_1d_model*")
        with open(data_path+'data_beats', 'rb') as f:
            beat_list = pickle.load(f)  
        with open(data_path+'labels_data_beats', 'rb') as f:
            target_values = pickle.load(f)      
    else:
        params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\LSTM_model*") 
        with open(data_path+'data_beats_LSTM', 'rb') as f:
            beat_list = pickle.load(f)  
        with open(data_path+'labels_data_beats_LSTM', 'rb') as f:
            target_values = pickle.load(f) 
            
elif data_type=="AF":
    data_path = r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_AF\\"
    if model_type=="CNN":
        if not conv_1d:
            params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\CNN_model*")
        else:
            params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\CNN_1d_model*")
        with open(data_path+'data_beats_singlescan_SR', 'rb') as f:
            beat_list = pickle.load(f)  
        with open(data_path+'labels_beat_singlescan_SR', 'rb') as f:
            target_values = pickle.load(f)      
    else:
        params_files = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\LSTM_model*") 
        with open(data_path+'data_beats_LSTM', 'rb') as f:
            beat_list = pickle.load(f)  
        with open(data_path+'labels_data_beats_LSTM', 'rb') as f:
            target_values = pickle.load(f) 
    
 

#%% GRAD CAM and OCCLUSION MAPS FOR THE WHOLE DATASET

        


path_models = r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\Fixed_Results_seed36PLN_CNN_200epochs"

with open(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\all_keys_pln.txt", "rb") as fp:   # Unpickling
    all_keys = pickle.load(fp)
all_keys  = np.array(all_keys)

init_seed = 36


l_rates = [0.00001,0.00005,0.0005]   
np.random.seed(init_seed)
splits = []
measures = []

skf = StratifiedKFold(n_splits=4, random_state=init_seed, shuffle=True)

epochs = 100

m_list = list()
m_eval = Measures()
tn=0
fp=0
fn=0
tp=0

pred_list = list()
split_keys = list()
y_test_list = list()

d=dict()
d[0] = 'conv_2'
d[1] = 'conv_2'
d[2] = 'conv_2'
d[3] = 'conv_2'

fold = 3

model = load_model(path_models+'\\CNN_model1.json1e-05_'+str(fold)+'.h5')

for fold,(train_index, test_index) in enumerate(skf.split(beat_list, target_values)):
    
    splits.append([test_index])
    split_keys=list(all_keys[test_index])
    
    train_index, val_index = \
        train_test_split(train_index, test_size=0.2, stratify=np.array(target_values)[train_index],random_state=init_seed)
    #reading json file with the parameters
                    
    #model = md.LSTM_model_1()
    
    if model_type == "CNN":
        X_tr,X_tes,X_val,y_tr,y_val,y_tes,input_shape = cd.Clean_Data_CNN(train_index, val_index,test_index,beat_list, target_values,False)                
    else:
        X_tr,X_tes,X_val,y_tr,y_val,y_tes = cd.Clean_Data_LSTM(train_index, val_index,test_index,beat_list, target_values)
        input_shape = (4,8,256,1)#LSTM
    #break
 
    for i in range(X_tes.shape[0]):    
        data = X_tes[i,:,:,:]
        y_t = y_tes[i,:]
        data = np.expand_dims(data,axis=0)
        
        y_t = y_tes[i,:]
    
        correct_class = np.argmax(y_t)
        # input tensor for model.predict
        inp = data
    
        # image data for matplotlib's imshow
        img = data.reshape(8, 256)
    
        # occlusion
        img_size = img.shape[0]
        occlusion_size = 3
    
        print('occluding...')
    
        heatmap = np.zeros((8, 256), np.float32)
        class_pixels = np.zeros((8, 256), np.int16)
    
        beg=0
        end=16
        
        for j in range(0,17):
            X = data.copy()
            X[:,:,beg:end,:] = 0  
        
            #X = np.expand_dims(X,axis=0)
        
            #X = img_float.reshape(1, 28, 28, 1)
            out = model.predict(X)
            #print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
            #print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))
            #when this is occlused, this is the probability that it is the right class
            heatmap[:, beg:end] = out[0][correct_class]
            class_pixels[:, beg:end] = np.argmax(out)
            beg=end
            end=end+16
          
        heatmap = heatmap/(np.max(heatmap))    
       
        
        lead_names = ['I','II','V1','V2','V3','V4','V5','V6']
        correct_order = [0,2,4,6,1,3,5,7]
        
    
               
        #trying fill betweeen
        gs1 = gridspec.GridSpec(6, 2,wspace=0.3,hspace=0.4)
        f = pylab.figure(figsize=(8,16))
        for p,j in enumerate(range(0,8)):
            ax = pylab.subplot(gs1[correct_order[j]])
            ax.set_ylim(-0.5,0.5)
            beg=0
            end=16
            y_pos = np.arange(0, 1, 1/256)
            y_pos2 = np.zeros(256)
            y_pos2 = y_pos2-0.5
            for k in range(1,17):
                x = X_tes[i,j,beg:end,0]
                y = y_pos[beg:end]
                y2 = y_pos2[beg:end] 
                val = np.mean(heatmap[j,beg:end])
                
                #ax.fill_between(y,x,y2, alpha=1-val,color='red')
                ax.plot(y,x,'r')
                beg = end
                end = end+16
            pylab.ylabel('voltage (%s)' % "mm")
            pylab.xlabel('Lead: %s' % lead_names[j])
        
        if str(split_keys[i]) in list_tn:
            plt.savefig(r"G:\diva1\Research\MUSE ECGs\PROJECTS\Hidde\PLN lucas hidde\Visualization\Occlusion_map\Fancy\all_but_better\TN\\"+str(split_keys[i])+'.jpg')
        else:
            if str(split_keys[i]) in list_tp:
                plt.savefig(r"G:\diva1\Research\MUSE ECGs\PROJECTS\Hidde\PLN lucas hidde\Visualization\Occlusion_map\Fancy\all_but_better\TP\\"+str(split_keys[i])+'.jpg')
            else:
                plt.savefig(r"G:\diva1\Research\MUSE ECGs\PROJECTS\Hidde\PLN lucas hidde\Visualization\Occlusion_map\Fancy\all_but_better\\"+str(split_keys[i])+'.jpg')


        
        #fig, axs = plt.subplots(4,2, figsize=(12, 8), facecolor='w', edgecolor='k')
        
#        gs1 = gridspec.GridSpec(6, 2,wspace=0.3,hspace=0.4)
#        f = pylab.figure(figsize=(8,12))
#        for p,j in enumerate(range(0,8)):
#            ax = pylab.subplot(gs1[correct_order[j]])
#            ax.set_ylim(-0.5,0.5)
#            ax.plot(np.arange(0, 1, 1/256),  X_tes[i,j,:,0],'r') 
#            y_pos = np.arange(0, 1, 1/256)
#            pylab.ylabel('voltage (%s)' % "mm")
#            pylab.xlabel('Lead: %s' % lead_names[j])
#           # plt.plot(y_pos,  X_tes[i,j,:,0],'-r') 
#            beg=0
#            end=25
#            s = np.ones(256)
#            for k in range(1,11):                   
#                val = np.mean(heatmap[j,beg:end])
#                s[beg:end] = 1- s[beg:end]*val 
#                beg = end
#                end = end+25
#            for k in range(1,256):
#                ax.axvspan(y_pos[k-1],y_pos[k] ,ymax= 0.5+X_tes[i,j,k,0]  ,color='red', alpha=s[k],edgecolor=None)    
#        
#        plt.savefig(r"G:\diva1\Research\MUSE ECGs\PROJECTS\Hidde\PLN lucas hidde\Visualization\Occlusion_map\Fancy\all\\"+str(split_keys[i])+'.jpg')
#        
 
        
    
       
#        for j in range(0,4):
#            temp = 221+j
#            #plt.figure( figsize=(4,2))
#            ax=plt.subplot(temp)
#            ax.set_ylim(-0.5,0.5)
#            ax.plot(np.arange(0, 1, 1/256),  X_tes[i,j,:,0],'r') 
#            y_pos = np.arange(0, 1, 1/256)
#           # plt.plot(y_pos,  X_tes[i,j,:,0],'-r') 
#            beg=0
#            end=25
#            s = np.ones(256)
#            for k in range(1,11):                   
#                val = np.mean(heatmap[j,beg:end])
#                s[beg:end] = s[beg:end]*val 
#                beg = end
#                end = end+25
#            for k in range(1,256):
#                ax.axvspan(y_pos[k-1],y_pos[k] ,ymax= 0.5+X_tes[i,j,k,0]  ,color='red', alpha=s[k-1])
#        plt.savefig(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\\Occlusion_maps_mixed\\"+str(split_keys[i])+'.jpg')
#        
        
    #GRad cam part    
#    model = load_model(path_models+'\\CNN_model1.json1e-05_'+str(fold)+'.h5')
#    y_pred = model.predict(X_tes)
#    
#    
#
#    for i in range(X_tes.shape[0]):
#        print(i)
#        X_v = X_tes[i,:,:,:]
#        y_t = y_tes[i,:]
#        X_v = np.expand_dims(X_v,axis=0)
#        y_p = model.predict(X_v)
#        y_p= np.argmax(y_p)
#        
#        gradcam,heatmap= grad_cam_mine(model,X_v,layer_name='conv_2',category_index=y_p)
#        
#        fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
#        fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
#        for j in range(0,8):
#            plt.figure( figsize=(4,2))
#            temp = 251+j
#            ax=plt.subplot(temp)
#            ax.set_ylim(-0.5,0.5)
#            ax.plot(np.arange(0, 10, 10/256),  X_tes[i,j,:,0])    
#            beg=0
#            end=25
#            s = np.ones(256)
#            
#            for k in range(1,11):                   
#                val = np.mean(heatmap[j,beg:end])
#                s[beg:end] = s[beg:end]*val 
#                ax.axvspan(k-1,k, color='green', alpha=val)
#                beg = end
#                end = end+25
#            for k in range(1,256):
#
#                 ax.axvspan(k-1,k,ymin=-0.5,ymax=X_tes[i,j,k-1,0], color='green', alpha=s[i])
#                
#        #plt.savefig(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\GRAD_CAM_mixed\\"+str(split_keys[i])+'.jpg')
#        plt.savefig(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\\Occlusion_maps_mixed\\"+str(split_keys[i])+'.jpg')
#    
    #pred_list.append(y_pred[:,0])
#    pred_list += list(y_pred[:,0])
#    #y_test_list.append(y_tes[:,0])    
#    y_test_list+=list(y_tes[:,0])    
#    
#    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_tes[:,1], (y_pred[:,1] > 0.5)).ravel()
#    tn += tn_t
#    fp += fp_t    
#    fn += fn_t
#    tp += tp_t
#    print(tn_t,fp_t,fn_t,tp_t)
#print(tn,fp,fn,tp)
#print(tn+tp)    
#print(fn+fp)    
#
#import pandas as pd
#
#frame = pd.DataFrame(columns = ['keys','prob','pred','y'])
#
#frame['keys'] = split_keys
#frame['prob'] = pred_list
#frame['y'] = y_test_list
#frame['pred'] = frame['prob'] >0.5
#frame['pred'] = frame['pred'].astype(int)
#
#tn_t, fp_t, fn_t, tp_t = confusion_matrix(frame['y'], (frame['pred'])).ravel()
#print('final',tn_t,fp_t,fn_t,tp_t)
#frame.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_PLN\predictions_per_patient.csv")    
   


#%% occlusion maps single model

from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2


import matplotlib.pylab as pylab

list_correct = list()
list_pred = list()

for i in range(0,X_tes.shape[0]):
    X_v = X_tes[i,:,:,:]
    y_t = y_tes[i,:]
    X_v = np.expand_dims(X_v,axis=0)
    y_p = model.predict(X_v)
    list_pred.append(y_p)
    y_p= np.argmax(y_p)
    if np.argmax(y_t)==1 and y_p==np.argmax(y_t):
#    if y_p==np.argmax(y_t):
        list_correct.append(i)


for c,i in enumerate(list_correct):
    data = X_tes[i,:,:,:]
    y_t = y_tes[i,:]
    data = np.expand_dims(data,axis=0)
    
    y_t = y_tes[i,:]

    correct_class = np.argmax(y_t)
    # input tensor for model.predict
    inp = data

    # image data for matplotlib's imshow
    img = data.reshape(8, 256)

    # occlusion
    img_size = img.shape[0]
    occlusion_size = 3

    print('occluding...')

    heatmap = np.zeros((8, 256), np.float32)
    class_pixels = np.zeros((8, 256), np.int16)

    beg=0
    end=25
    
    for j in range(0,11):
        X = data.copy()
        X[:,:,beg:end,:] = 0  
    
        #X = np.expand_dims(X,axis=0)
    
        #X = img_float.reshape(1, 28, 28, 1)
        out = model.predict(X)
        #print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
        #print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))
    
        heatmap[:, beg:end] = out[0][correct_class]
        class_pixels[:, beg:end] = np.argmax(out)
        beg=end
        end=end+25
        
    heatmap = heatmap/(np.max(heatmap))    
    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    
    for j in range(0,8):
        temp = 251+j
        ax=plt.subplot(temp)
        ax.set_ylim(-0.5,0.5)
        ax.plot(np.arange(0, 10, 10/256), X_tes[i,j,:,0])    
        beg=0
        end=25
        for k in range(1,11):                   
            val = np.mean(heatmap[j,beg:end])
            ax.axvspan(k-1,k, color='green', alpha=val)
            beg = end
            end = end+25
        plt.savefig(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\Occlusion_maps _results\\"+str(i)+'.jpg')
        
  


#%%GRADCAM single model

from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2

list_correct = list()
list_pred = list()
for i in range(0,X_tes.shape[0]):
    X_v = X_tes[i,:,:,:]
    y_t = y_tes[i,:]
    X_v = np.expand_dims(X_v,axis=0)
    y_p = model.predict(X_v)
    list_pred.append(y_p)
    y_p= np.argmax(y_p)
    if np.argmax(y_t)==1 and y_p==np.argmax(y_t):
#    if y_p==np.argmax(y_t):
        list_correct.append(i)

      

            
from grad_cam import grad_cam_mine


for c,i in enumerate(list_correct):
    X_v = X_tes[i,:,:,:]
    y_t = y_tes[i,:]
    X_v = np.expand_dims(X_v,axis=0)
    y_p = model.predict(X_v)
    y_p= np.argmax(y_p)
    #y_p= int(y_p[0,0]>0.5)
    
    #gradcam = gradcamutils.grad_cam(model,X_v,layer_name='conv3',W=8,H=256)
    gradcam,heatmap= grad_cam_mine(model,X_v,layer_name='conv_2',category_index=y_p)
    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
   
    
    for j in range(0,8):
        temp = 251+j
        ax=plt.subplot(temp)
        ax.plot(np.arange(0, 10, 10/256), X_tes[i,j,:,0])
    
        beg=0
        end=25
        for k in range(1,11):                   
            val = np.mean(heatmap[j,beg:end])
            ax.axvspan(k-1,k, color='green', alpha=val)
            beg = end
            end = end+25
    plt.savefig(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\Grad_cam_results\\"+str(i)+'.jpg')
    
    #    plt.plot(np.arange(0, 10, 10/256), gradcam[j,:])
    #    plt.show()
        
           