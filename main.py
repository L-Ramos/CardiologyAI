# -*- coding: utf-8 -*-
"""
This code was created by Ricardo and Lucas to parse ECG files in the .xml format

The code contains functions to load all files and map them to their IDS, save and plot them.

The names of the fields can change depending on the database, for the MUSE it is already set

If you have any questions feel free to contact us on:

r.riccilopes@amc.uva.nl Ricardo R Lopes or
l.a.ramos@amc.uva.nl - Lucas A. Ramos



To filter all the ECGs in Sinus rhythm out of the dataset of the selected AF patients who underwent AF surgery, we can use the following part of the XML. 

<DiagnosisStatement><StmtFlag>ENDSLINE</StmtFlag><StmtText>Sinusbradycardie</StmtText>

For correct filtering we have to use all the synonyms for sinus rhythm (in dutch): Sinusritme, Sinus ritme, Sinusbradycardie, Sinustachycardie. 

Best, Sarah 


"""


from aux_functions import plot_signal, get_xml_hash, get_excel_from_xml, filter_by_date, save_file, load_file, fix_excel, filter_rhythm
import pandas as pd
import os
import numpy as np
import base64
import tqdm #need to install
from collections import defaultdict
import xmltodict #need to install

xml_path=r"G:\diva1\Research\MUSE ECGs\**\*.xml" #G disk folder with all ecgs
df = pd.read_excel(r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\Sarah outcome MiniMaze_DL.xlsx") #Excel file to use for filtering the ecgs

#-------------------------------------------------------------------------------------------------------
# DON'T FORGET TO SET THE PATH BELOW
#-------------------------------------------------------------------------------------------------------

#path where you want to save or load the processed files, if you already have a hash table and data files this is the path to it
path_files=r"\\amc.intra\users\L\laramos\home\Desktop\Cardio ECG\\final"

#some names for the files, hash table is called ids_and_paths, error_files are the xml with problems for parsing, ecg_signal is dicionary with keys and signals
#dates is a dictionary with the date of acquisition of each ecg and amplitudes for each ecg amplitude (sanity check)
files_names= ['\\ids_and_paths','\\error_files','\\ecg_signal','\\dates','\\amplitudes']

excel_field='PIN' #name of Excel field of patient ID, in this case it is the PIN
xml_field="PatientID" #.xml field to connect to excel_field
excel_surgery="Surgery Date" # name of the date field from the excel


#fixing excel problems, change according to your file
excel_ids,excel_dates=fix_excel(df,excel_field,excel_surgery)
        
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
    data_signal=load_file(path_files,files_names[2])  
    date=load_file(path_files,files_names[3])  
    data_amp=load_file(path_files,files_names[4])  

#----------------------------------------------------------------------------------
#Filter for date   
#----------------------------------------------------------------------------------
#filter_date contains all the xml that are before surgery date
print("Filtering based on date!")
filter_date,later_date=filter_by_date(hash_table,excel_ids,excel_dates,True)#True=only < 6 months
print("Done filtering date!")
                    
#Now we filter based on rhythm, sinus in this case
print("Filtering based on rhythm")
filter_rhythm,other_rhythms=filter_rhythm(filter_date)
print("Done!") 


#Final get's the signal
data_signal = defaultdict(list)  

for key in tqdm(filter_rhythm): 
        file_names=filter_rhythm[key]
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

for key in tqdm(filter_rhythm): 
    signal=data_signal[key]    
    for i in range(len(filter_rhythm[key])):
        file_name=filter_rhythm[key][i]        
        plot_signal(np.array(signal[i*8]),os.path.basename(file_name),path_files)





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
                