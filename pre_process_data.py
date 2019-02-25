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

def plot_signal(signal,file_name,path_files):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)

    # TODO: Fix grid
    if False:
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0))
        ax.grid(which='major', axis='x', linewidth=2, alpha=0.3, linestyle='-', color='red')
        ax.grid(which='minor', axis='x', linewidth=0.5, alpha=0.2, linestyle='-', color='red')
        ax.grid(which='major', axis='y', linewidth=2, alpha=0.3, linestyle='-', color='red')
        ax.grid(which='minor', axis='y', linewidth=0.5, alpha=0.2, linestyle='-', color='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])


    max_len = int(len(signal)/2)
    
    plt.plot(np.arange(0, 5, 5/max_len), signal[:max_len])
    plt.savefig(path_files+'\\images\\'+file_name+'.png')
    plt.close(fig) 
    #plt.show()

def get_xml_hash(xml_path):
    """
    This function gets all the xml files in a given path and write down the patientID and the path to the xml file
    This way we don't have to search for the patients every time
    Input:
        xml_path: string, path to all the xml files, if you have several folder use ** for the different folders as shown in the example
    Output:
        hash_table: dictionary with patientid as key and a list of paths for each id, this has to ve saved outside the function

    """
    hash_table = defaultdict(list)
    print("Catching all files!")
    file_list=glob.glob((xml_path), recursive=True)
    
    #file_list=glob.glob((r"G:\diva1\Research\MUSE ECGs\00_01_02_03\*.xml"), recursive=True)
    error_table = list()
    for file in tqdm(file_list):
    #for file in file_list:
        with open(file) as fd:
            try:
                ecg_dict = xmltodict.parse(fd.read(), process_namespaces=True)                        
                f_id = ecg_dict['RestingECG']['PatientDemographics'][xml_field]
                #all_xml[f_id].append(file)
                hash_table[f_id].append(file)
            except:
               # print("Error:",file)
                error_table.append(file)
                
    return(hash_table,error_table)
    
def get_excel_from_xml(excel_ids,hash_table):
    """
    This function loads the ecg signals from the xml files according to the excel file
    Input:
        excel_ids: list of all the ids from the excel file (PIN)
        hash_table: output from get_xml_hash
    Output: 
        data_signal: dictionary with patient ID and the loaded signal
        date:  dictionary with date of acquisition of the ecg
        data_amp: dictionary with amplitudes just to make sure they are not changing = (to be removed)
    """
    
    data_signal = defaultdict(list)        
    date = defaultdict(list) 
    data_amp = defaultdict(list) 
    for key in tqdm(excel_ids): 
        file_names=hash_table[key]
        for i in range(len(file_names)): 
            with open(file_names[i]) as fd:
                ecg_dict = xmltodict.parse(fd.read(), process_namespaces=True)
            mean_wave, leads = ecg_dict['RestingECG']['Waveform']
            date[key].append(ecg_dict['RestingECG']['TestDemographics']['AcquisitionDate'])          
            for k,i in enumerate(leads['LeadData'][:]):
                amp=float(leads['LeadData'][k]['LeadAmplitudeUnitsPerBit'])
                data_amp[key].append(amp)
                b64_encoded = ''.join(i['WaveFormData'].split('\n'))
                decoded = base64.b64decode(b64_encoded)
                signal = np.frombuffer(decoded, dtype='int16')
                data_signal[key].append(signal*amp)
                #plot_ecg(signal)    
    return(data_signal,date,data_amp)
    
def filter_by_date(hash_table,excel_ids,excel_dates,six_months):
    """
    This function will further filter patients based on the date read from the file (excel in this case)
    Current options are 6 months before surgery or just before surgery, but that can be easily adapted.
    Input:
        excel_ids: list of all the ids from the excel file (PIN)
        excel_dates: dictionary with the keys from each patient ID and the date to be filtered (surgery date in this case)
        six_months: boolean, if True, only keeps the ECGs that were done 6 months before surgery date
        
    Output:
        filter_date: Dictionary with a list of .xml paths for each key whose date was < than surgery date
        later_date: same of filter_date but the ones that did not fit the search criterea        
    """
    filter_date = defaultdict(list)  
    
    later_date = defaultdict(list)  
    
    found=False
    #number of days in 6 months
    days_in_months=183
    
    for key in (excel_ids):        
            file_names=hash_table[key]
            file_d=date[key]
            excel_d=datetime.strptime(excel_dates[key], "%d-%m-%Y")
            for i in range(len(file_d)):
                d=datetime.strptime(file_d[i], "%m-%d-%Y")        
                if six_months:
                    if (excel_d-d).days<days_in_months and (excel_d-d).days>0:                 
                        filter_date[key].append(file_names[i])
                        found=True
                else:
                    if d<excel_d:                 
                        filter_date[key].append(file_names[i])
                        found=True
            if not found:
                later_date[key].append(file_names)
            else:
                found=False
                
    return(filter_date,later_date)

def save_file(table_save,path_save,file_name):    
    filename = path_save+file_name
    outfile = open(filename,'wb')
    pickle.dump(table_save,outfile)
    outfile.close()
    
def load_file(path_save,filename):
    infile = open(path_save+filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return(new_dict)
    
def fix_excel(df,excel_field,excel_surgery):
    """
    This function fixes some errors that came with the excel file, probably not very useful if you switch files, but can be adapted and makes the code easier to read
    Input:
        df: Pandas Dataframe from the excel file
        excel_field: name of Excel field of patient ID, in this case it is the PIN
        excel_surgery: name of Excel field of Surgery Date
    Output:
        excel_ids:  list of all the ids from the excel file (PIN) after corrections
        excel_dates: dictionary with the keys from each patient ID and the date to be filtered (surgery date in this case)
    """
    
    excel_ids=list(df[excel_field].astype(str))
    #add zeros because the number should have 7 digits
    for i in range(len(excel_ids)):
        if len(excel_ids[i])<7:
            excel_ids[i]=excel_ids[i].rjust(7,'0')
            
        
    excel_dates=list(df[excel_surgery].astype(str))
    
    #Replacing nans with ridiculous date instead of excluding, they will never be used anyway
    excel_dates = ['1-1-1500' if str(x)=='nan' else x for x in excel_dates]  
    excel_dates = dict(zip(excel_ids,excel_dates))
    
    return(excel_ids,excel_dates)

def filter_rhythm(filter_date):
    filter_rhythm=defaultdict(list)   
    other_rhythms=defaultdict(list) 
    for key in tqdm(filter_date): 
        file_names=filter_date[key]  
        for file in file_names:       
            with open(file) as fd:
                    rhythm = str(xmltodict.parse(fd.read(), process_namespaces=True))
                    if ('Sinusritme'in rhythm or 'Sinus ritme'in rhythm or 'Sinusbradycardie'in rhythm or 'Sinustachycardie' in rhythm):
                        filter_rhythm[key].append(file)
                    else:
                        other_rhythms[key].append(rhythm)
    return(filter_rhythm,other_rhythms)
                        

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
                