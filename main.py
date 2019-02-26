# -*- coding: utf-8 -*-
"""
This code was created by Ricardo and Lucas to parse ECG files in the .xml format

The code contains functions to load all files and map them to their IDS, save and plot them.

The names of the fields can change depending on the database, for the MUSE it is already set

If you have any questions feel free to contact us on:

r.riccilopes@amc.uva.nl Ricardo R Lopes or
l.a.ramos@amc.uva.nl - Lucas A. Ramos

"""


from aux_functions import plot_signal, get_xml_hash, get_excel_from_xml, filter_by_date, save_file, load_file, fix_excel, filter_by_rhythm
import aux_functions as af
import xmltodict #need to install
import base64
import numpy as np
import matplotlib.pyplot as plt
import glob
import os 
import pandas as pd
from collections import defaultdict
from tqdm import tqdm #need to install
import pickle #need to install
from datetime import datetime

xml_path=r"G:\diva1\Research\MUSE ECGs\**\*.xml" #G disk folder with all ecgs

#-------------------------------------------------------------------------------------------------------
# DON'T FORGET TO SET THE 2 PATHs BELOW
#-------------------------------------------------------------------------------------------------------

#path where you want to save or load the processed files, if you already have a hash table and data files this is the path to it
path_files=r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\\final"
df = pd.read_excel(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\Sarah outcome MiniMaze_DL.xlsx") #Excel file to use for filtering the ecgs

#some names for the files, hash table is called ids_and_paths, error_files are the xml with problems for parsing, ecg_signal is dicionary with keys and signals
#dates is a dictionary with the date of acquisition of each ecg and amplitudes for each ecg amplitude (sanity check)
files_names= ['\\ids_and_paths','\\error_files','\\ecg_signal','\\dates','\\amplitudes']

excel_field='PIN' #name of Excel field of patient ID, in this case it is the PIN
xml_field="PatientID" #.xml field to connect to excel_field
excel_surgery="Surgery Date" # name of the date field from the excel
label_name="Out_1Y"

#fixing excel problems, change according to your file
excel_ids,excel_dates,labels=fix_excel(df,excel_field,excel_surgery,label_name)
        
#Checking if you already have the hash table, if you don't you should ask for it, otherwise this will take around 12 hours to run    
if not os.path.isfile(path_files+files_names[0]):
    print("Hash table not found, check path or wait till one is created. Estimated time is shown below:")
    hash_table,error_files=get_xml_hash(xml_path)
    save_file(hash_table,path_files,files_names[0])
    save_file(error_files,path_files, files_names[1])
else:
    print("Hash table already available, loading!")
    hash_table=load_file(path_files,files_names[0])    
    error_files=load_file(path_files,files_names[1])    


#------------------------------------------------------------------------------------
#Filters by excel ids, if you need other filters check bellow
#------------------------------------------------------------------------------------
#Check if the signal file has already been created, otherwise created it, this is where we filter the data based on the excel
if not os.path.isfile(path_files+files_names[2]):
    print("Previous file not found. Matching Excel with .xml:")
    data_signal,date,data_amp=get_excel_from_xml(excel_ids,hash_table)
    save_file(data_signal,path_files,files_names[2])
    save_file(date,path_files, files_names[3])
    save_file(data_amp,path_files,files_names[4])
else:
    print("Loading Data!")
    #data_signal=load_file(path_files,files_names[2])  
    date=load_file(path_files,files_names[3])  
    #data_amp=load_file(path_files,files_names[4])  

#----------------------------------------------------------------------------------
#Filter for date   
#----------------------------------------------------------------------------------
#filter_date contains all the xml that are before surgery date
print("Filtering based on date!")
filter_date,later_date=filter_by_date(hash_table,excel_ids,excel_dates,date,True)#True=only < 6 months
print("Done filtering date!")
                    
#Now we filter based on rhythm, sinus in this case
print("Filtering based on rhythm")
filter_rhythm,other_rhythms=filter_by_rhythm(filter_date)
print("Done!") 


#Final get's the signal
data_signal = defaultdict(list)  
final_labels = defaultdict(list)  
for key in tqdm(filter_rhythm): 
    if labels[key]!='999.0' and labels[key]!='nan':
        file_names=filter_rhythm[key]
        final_labels[key].append(labels[key])        
        for i in range(len(file_names)): 
            with open(file_names[i]) as fd:
                ecg_dict = xmltodict.parse(fd.read(), process_namespaces=True)
            mean_wave, leads = ecg_dict['RestingECG']['Waveform']                
            for k,i in enumerate(leads['LeadData'][:]):
                amp=float(leads['LeadData'][k]['LeadAmplitudeUnitsPerBit'])               
                b64_encoded = ''.join(i['WaveFormData'].split('\n'))
                decoded = base64.b64decode(b64_encoded)
                signal = np.frombuffer(decoded, dtype='int16')
                data_signal[key].append(signal*amp)
                #plot_ecg(signal)    

 

#plotting and saving for visual analysis
for key in tqdm(filter_rhythm): 
    signal=data_signal[key]    
    for i in range(len(filter_rhythm[key])):
        file_name=filter_rhythm[key][i]        
        plot_signal(np.array(signal[i*8]),os.path.basename(file_name),path_files)


#quick check label balance and cout total leads
cont_labels=0
total_leads=0
for key in final_labels:
    cont_labels+=float(final_labels[key][0])
    total_leads+=len(data_signal[key])  
    
#total_leads=int(total_leads/8)
    
#Fixing it to a nice format, numpy FTW
keys=list(data_signal.keys())
final_data=np.zeros((5000,total_leads),dtype="float32")
cont_lead=0
for i,key in enumerate(final_labels):
    leads=data_signal[key]
    
    #for l in range(0,len(leads),8):
    for l in range(0,len(leads)):
        if leads[l].shape[0]==5000:
            final_data[:,cont_lead]=leads[l]        
            cont_lead+=1
        else:
            final_data[0:2500,cont_lead]=leads[l]        
            cont_lead+=1
    
    
for i,key in enumerate(final_labels):
    leads=data_signal[key]
    found=False
    for l in range(0,len(leads),8):
        if leads[l].shape[0]==5000:
            found=True
    if not found:
        print(key)




total_ecgs=0
for key in (hash_table):
    total_ecgs+=len(hash_table[key])
print("Total ECGs in the original folder:%d, total patients: %d"%(total_ecgs,len(hash_table.keys())))

total_ecgs=0
for key in (filter_date):
    total_ecgs+=len(filter_date[key])    
print("Total ECGs after filtering date:%d, total patients: %d"%(total_ecgs,len(filter_date.keys())))
                

total_ecgs=0
for key in (filter_rhythm):
    total_ecgs+=len(filter_rhythm[key])    
print("Total ECGs after filtering by rhythm:%d, total patients:%d "%(total_ecgs,len(filter_rhythm.keys())))
                