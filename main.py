
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from numpy.random import seed
from tensorflow import set_random_seed
import os
from sklearn.metrics import f1_score

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers  

#from imblearn.under_sampling import RandomUnderSampler
import pickle
import json
import glob
import ntpath

#built json models
import models as md
import clean_data as cd
from sklearn.linear_model import LogisticRegression

 


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
    
 

# %% ----------------------------------------------------------------------
### Machine learning
# -*- coding: utf-8 -*-

l_rates = [0.00001,0.00005,0.0005]   
np.random.seed(init_seed)
splits = []
measures = []

skf = StratifiedKFold(n_splits=4, random_state=init_seed, shuffle=True)

epochs = 100

m_list = list()
    
for file_name in params_files:
    for lr in l_rates:
        m_eval = Measures()
        for fold,(train_index, test_index) in enumerate(skf.split(beat_list, target_values)):
            
            splits.append([test_index])
        
            train_index, val_index = \
                train_test_split(train_index, test_size=0.2, stratify=np.array(target_values)[train_index],random_state=init_seed)
            #reading json file with the parameters
            name = ntpath.basename(file_name)
            
            #loads the parameters
            with open(file_name, 'r') as f:
                 params = json.load(f)
                    
            #fixed model
            #model = md.LSTM_model_1()
            
            if model_type == "CNN":
                X_tr,X_tes,X_val,y_tr,y_val,y_tes,input_shape = cd.Clean_Data_CNN(train_index, val_index,test_index,beat_list, target_values,conv_1d)                
            else:
                X_tr,X_tes,X_val,y_tr,y_val,y_tes = cd.Clean_Data_LSTM(train_index, val_index,test_index,beat_list, target_values)
                input_shape = (4,8,256,1)#LSTM
            if fold>0:
                break

            
            sample_size = y_tr.shape[0]
            tot_pos = np.sum(y_tr,axis=0)
            tot_neg = sample_size - tot_pos

            w_0 = (1 / tot_neg)*(sample_size)/2.0 
            w_1 = (1 / tot_pos)*(sample_size)/2.0
            
            c_w_0 = {0: w_0, 1:w_1}  
            
            model = md.Build_Model(params,input_shape,lr,w_0,w_1)  
            #model = Cnn_Model_1()
            rms = optimizers.RMSprop(lr=lr, decay=1e-6)
            #rms = optimizers.Adam(lr=lr)
            model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    
            history = model.fit(X_tr,
                                  y_tr,
                                  #batch_size=int(X_tr.shape[0]/4),#Weird, too big
                                  batch_size=128,
                                  shuffle=True,
                                  epochs=epochs,
                                  #validation_split=0.2,
                                  validation_data=(X_val, y_val),
                                  verbose=1)
                                  #class_weight=[c_w_0])
            #models.append(model)
            #model_cam = model
            # summarize history for accuracy
            plt.figure()
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model acc')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(results_folder+'acc_train_validation_'+name+str(lr)+'.pdf')
            
            # summarize history for loss
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(results_folder+'loss_train_validation_'+name+str(lr)+'.pdf')
        
            y_pred = model.predict(X_tes)
            m_eval.predictions.append(y_pred)
            m_eval.labels.append(y_tes)
            tn, fp, fn, tp = confusion_matrix(y_tes[:,1], (y_pred[:,1] > 0.5)).ravel()   
            
            m_eval.sens.append(tp/(tp+fn))
            m_eval.spec.append(tn/(tn+fp))
            m_eval.acc.append(accuracy_score(y_tes[:,1], (y_pred[:,1] > 0.5)))
            m_eval.f1.append(f1_score(y_tes[:,1], (y_pred[:,1] > 0.5)))
            print("Sensitivity: ",tp/(tp+fn))  
            print("Specificity: ",tn/(tn+fp))
            print("Accuracy: ",accuracy_score(y_tes[:,1], (y_pred[:,1] > 0.5)))
            
            y_pred_tr = model.predict(X_tr)
            y_pred_tes = model.predict(X_tes)
            
            fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr[:,1], y_pred_tr[:,1])
            auc_tr = auc(fpr_tr, tpr_tr)
            fpr_tes, tpr_tes, thresholds_tes = roc_curve(y_tes[:,1], y_pred_tes[:,1])
            auc_tes = auc(fpr_tes, tpr_tes)
            m_eval.auc.append(auc_tes)
            
            plt.figure(figsize=(7,6))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_tes, tpr_tes, label='Testing = {:.2f}'.format(auc_tes))
            plt.plot(fpr_tr, tpr_tr, label='Training = {:.2f}'.format(auc_tr))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('Receiver-operating characteristic curve', fontsize=13)
            plt.legend(loc='lower right')
            plt.savefig(results_folder+'auc_'+name+str(lr)+'.pdf')    
            model.save(results_folder+name+str(lr)+"_"+str(fold)+'.h5')
            #clear the model, to make sure it trains a new one, prevents data leakage
            del model
            tf.reset_default_graph() # for being sure
            K.clear_session()
            #cuda.select_device(0)
            #cuda.close()
        m_list.append(m_eval)
        print(m_eval.auc)
        

      
with open(results_folder+'measures'+data_type+"_"+model_type, 'wb') as f:
    pickle.dump(m_list,f)

#prints some results        
for m in m_list:
    print(np.mean(m.acc))
    
for m in m_list:
    print(np.mean(m.auc))
    

  