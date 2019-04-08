# -*- coding: utf-8 -*-
"""
This code was created by Ricardo and Lucas to parse ECG files in the .xml format

The code contains functions to load all files and map them to their IDS, save and plot them.

The names of the fields can change depending on the database, for the MUSE it is already set

If you have any questions feel free to contact us on:

r.riccilopes@amc.uva.nl Ricardo R Lopes or
l.a.ramos@amc.uva.nl - Lucas A. Ramos






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

def plot_signal(signal_list,file_name,path_files):
    
    n_plots=len(signal_list)
        
    fig = plt.figure(figsize=(35, 15))
    ax = fig.add_subplot(n_plots, 1, 1)
    
    x=211
    
    for i in range(n_plots):
        s=np.array(signal_list[i])
        
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


        max_len = int(s.shape[0]/2)
        plt.subplot(x)
        x=x+1
        plt.plot(np.arange(0, 5, 5/max_len), s[:max_len])
        #plt.show()
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
    
def filter_by_date(hash_table,excel_ids,excel_dates,date,six_months):
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
    
def fix_excel(df,excel_field,excel_surgery,label_name):
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
    excel_labels=list(df[label_name].astype(str))
    
    #Replacing nans with ridiculous date instead of excluding, they will never be used anyway
    excel_dates = ['1-1-1500' if str(x)=='nan' else x for x in excel_dates]  
    excel_dates = dict(zip(excel_ids,excel_dates))
    labels = dict(zip(excel_ids,excel_labels))
    
    return(excel_ids,excel_dates,labels)

def filter_by_rhythm(filter_date):
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
