# -*- coding: utf-8 -*-
"""
This code is for splitting the data into beats and organizing it for 1D, 2D Cnns and LSTMS

@author: laramos
"""
import base64
import xmltodict #need to install
from collections import defaultdict
import numpy as np
import scipy.signal as sps
from aux_ecg import get_patient_beats_nleads, get_beat_idx, resample_patient_signal, check_inverted_lead_one, statistics_len_beats
import pickle

#%%
def Read_Data(path,n_leads): 
    #Path: all the pathes from the ecgs
    #n_leads: total leads in each ecg
    control = defaultdict(list) 
    for key in (path): 
        file_names=path[key]
        if len(file_names[0][0])>1:
            file_names = file_names[0]
        with open(file_names[0]) as fd:
            ecg_dict = xmltodict.parse(fd.read(), process_namespaces=True)
            mean_wave, leads = ecg_dict['RestingECG']['Waveform']
            for k,i in enumerate(leads['LeadData'][:]):
                amp=float(leads['LeadData'][k]['LeadAmplitudeUnitsPerBit'])                
                b64_encoded = ''.join(i['WaveFormData'].split('\n'))
                decoded = base64.b64decode(b64_encoded)
                s = np.frombuffer(decoded, dtype='int16')
                control[key].append(s*amp)                
    return(control)
    
def Resample_Data(waves,resample_num):
    #waves: deault dictionary returned by Read_Data
    #resample_num: new lead size
    waves = list(waves.values())
    wave_list = []
    for patient in waves:
        temp_lead = []
        for lead in patient:
            res = sps.resample(lead, resample_num)
            temp_lead.append(res)
        wave_list.append(temp_lead)
    return(wave_list)
    
def Extract_Beats(wave_list,beat_size,n_leads,max_beats=0): 
    #wave_list:  list of lists of leads
    #beat_Size: size ot rescale a beat to
    #max_beats: max of beats per leads to take, important for methods that use multiple beats like the LSTM
    total_beat_list = []
    for sample in wave_list:
        # get beat index from first lead
        try:
            _, beat_idx = get_beat_idx(sample[0])
            beat_list = []        
            if max_beats==0:
                max_val = len(beat_idx)
            else:
                max_val = max_beats
            for b in range(0,max_val):
                beat=beat_idx[b] 
                temp_beat_list = []
                for lead in sample[:n_leads]:
                    res = sps.resample(lead[beat[0]:beat[1]], beat_size)
                    temp_beat_list.append(res)
                beat_list.append(temp_beat_list)
            total_beat_list.append(beat_list)
            print(len(total_beat_list))
        except:
            print("No peaks!")            
    return(total_beat_list)
 

    
#%%---------------------------- PLN ------------------------------------
    
n_leads = 8
beat_size = 256
resample_num = 2500

paths_p = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\Hidde\data\PLN_final_data_paths_18-60",allow_pickle=True) 
pln = Read_Data(paths_p,n_leads)

paths_c = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\Hidde\data\control_group_matched_plot_18-60",allow_pickle=True)
#paths_c = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\Hidde\data\control_group_paths_complete_18-60",allow_pickle=True)
control = Read_Data(paths_c,n_leads)

keys_pln = list(pln.keys())
keys_control = list(control.keys())

all_keys = keys_pln + keys_control

with open(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\all_keys_pln.txt", "wb") as fp:   
    pickle.dump(all_keys, fp)

pln_list = Resample_Data(pln,resample_num)
control_list = Resample_Data(control,resample_num)


data = control_list + pln_list
labels = len(control_list) * [0] + len(pln_list) * [1]

with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\data_imbalanced', 'wb') as f:
    pickle.dump(data,f)

with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\labels_data_imbalanced', 'wb') as f:
    pickle.dump(labels,f)  

#pln_list = pln
#control_list = control

del pln, control
 
# Extracting beats from lead 1 and using that for the others
pln_beat_list = Extract_Beats(pln_list,beat_size,n_leads)
control_beat_list = Extract_Beats(control_list,beat_size,n_leads)

#pln_beat_list_lstm = Extract_Beats(pln_list,beat_size,n_leads,max_beats=4)
#control_beat_list_lstm = Extract_Beats(control_list,beat_size,n_leads,max_beats=4)

data = control_beat_list + pln_beat_list
labels = len(control_beat_list) * [0] + len(pln_beat_list) * [1]

with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\data_beats_imbalanced', 'wb') as f:
    pickle.dump(data,f)

with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\labels_data_beats_imbalanced', 'wb') as f:
    pickle.dump(labels,f)   
    
    
#data = control_beat_list + pln_beat_list
#labels = len(control_beat_list) * [0] + len(pln_beat_list) * [1]    

#with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\data_beats', 'wb') as f:
#    pickle.dump(data,f)
#
#with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\data\labels_data_beats', 'wb') as f:
#    pickle.dump(labels,f)    
    

#%% -------------------------------------------------------- AF --------------------------------------------------------------



    
n_leads = 8
beat_size = 256
resample_num = 2500

data_path = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_AF\data_path_SR_date_filter",allow_pickle=True) 
data = Read_Data(data_path,n_leads)

label = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_AF\label_path_SR_date_filter",allow_pickle=True) 

for k in data_path:
    if label[k]==[]:
        del (data[k])
        del (label[k])
        
        
data_list = Resample_Data(data,resample_num)
data_beats = Extract_Beats(data_list,beat_size,n_leads)

label_list  = list()

for k in label:
        label_list.append(int(label[k][0]))        
    

with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_AF\data_beats_singlescan_SR', 'wb') as f:
    pickle.dump(data_beats,f)
    
with open(r'\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\ML pipeline\Latest Code\data_AF\labels_beat_singlescan_SR', 'wb') as f:
    pickle.dump(label_list,f)    
    

#%%
#Checking samples per labels
total_pos = 0
total_neg = 0
for i in range (0,len(data_beats)):
    if label_list[i]==1:
        total_pos+=(len(data_beats[i]))
    else:
        total_neg+=(len(data_beats[i]))
        
    


total_pos = 0
total_neg = 0
for i in range (0,len(beat_list)):
    if target_values[i]==1:
        total_pos+=(len(beat_list[i]))
    else:
        total_neg+=(len(beat_list[i]))
            