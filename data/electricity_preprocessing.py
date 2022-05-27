import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import quantile_transform
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
import datetime 
import math 
import random

'''
get_processed_dataset:
--get_dataset:
----read csv file and drop FLAGS
----sort date
----check missing date and add empty data for the missing date
----add FLAGS back
----user_based_preprocessing:
------All NaN: return False
------For mask:
--------NaN:0; Normal:1
------For elec data:
--------NaN: (elec[i-1] + elec[i+1]) / 2.0 or 0.0
--------Outlier: elec_mean + 2*elec_std
--------MinMaxScaler
--------elec = np.stack((norm_elec, mask_elec))
------meta=np.stack((meta_elec_max, meta_elec_min, meta_elec_mean, meta_mask_mean))
----np.savez: elec, meta, flag
'''

def download_data():
    # Download data from GitHub repository
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip')

    # Unzip downloaded data
    os.system('cat data.z01 data.z02 data.zip > data_compress.zip')
    os.system('unzip -n -q data_compress')


def get_dataset(filepath):
    """## Saving "flags" """

    df_raw = pd.read_csv(filepath,index_col=0)
    flags = df_raw.FLAG.copy()
 
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Sorting"""
    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)
    print('before check_missing_date')
    print(df_raw)

    ### add missing date ###
    print('\ncheck_missing_date')
    miss_date = check_missing_date(df_raw.index.values)

    # for date in miss_date:
    #    print('date adding:', date)
    #
    #    # df_raw.loc[date] = {}

    df_raw = df_raw.reindex(df_raw.index.union(pd.DatetimeIndex(miss_date, dtype="datetime64[ns]")))

    print('after check_missing_date')
    print(df_raw)
    df_raw.sort_index(inplace=True, axis=0)
    
    print('\nresort date')
    print(df_raw)
    ########################

    df_raw = df_raw.T.copy()
    df_raw['FLAG'] = flags
    return df_raw

def check_missing_date(date_list):
    date_min = date_list[0]
    date_max = date_list[-1]
    print('min date:', date_min)
    print('max date:', date_max)
    
    date_list= list(date_list)
    cur_date = date_min
    miss_date = []
    while(cur_date <= date_max):
       if cur_date not in date_list:
          print('middle date missing:', cur_date)
          miss_date.append(cur_date)

       cur_date = pd.Timestamp(cur_date) + datetime.timedelta(days=1)
       cur_date = cur_date.to_datetime64()
       
    cur_date = date_max
    while((len(date_list)+len(miss_date)) % 7 != 0):
       cur_date = pd.Timestamp(cur_date) + datetime.timedelta(days=1)
       cur_date = cur_date.to_datetime64()
       miss_date.append(cur_date)
       print('terminal date missing:', cur_date)
    
    return miss_date

def user_based_preprocessing(elec):
    
    mask_elec =  1.0 - np.isnan(elec)
    valid_elec = elec[np.isnan(elec) == False]
    
    if valid_elec.size == 0:
       return False, False, False
    
    elec_mean = np.mean(valid_elec)
    elec_std = np.std(valid_elec)

    norm_elec = elec.copy()
    for i in range(elec.shape[0]):
       if np.isnan(elec[i]):
          if i-1>=0 and np.isnan(elec[i-1])==False and i+1<elec.shape[0] and np.isnan(elec[i+1])==False:
              norm_elec[i] = (elec[i-1] + elec[i+1]) / 2.0
          else:
              norm_elec[i] = 0.0
       else:
          if elec[i] > elec_mean + 2*elec_std:
             norm_elec[i] = elec_mean + 2*elec_std
          else:
             norm_elec[i] = elec[i]
             
    norm_elec = norm_elec.reshape([-1, 7])
    mask_elec = mask_elec.reshape([-1, 7])  
    
    meta_elec_max  = np.max(norm_elec, 1).reshape([-1, 1])
    meta_elec_min  = np.min(norm_elec, 1).reshape([-1, 1])
    meta_elec_mean = np.mean(norm_elec, 1).reshape([-1, 1])
    meta_mask_mean = np.mean(mask_elec, 1).reshape([-1, 1])
    meta = np.stack((meta_elec_max, meta_elec_min, meta_elec_mean, meta_mask_mean))
           
    #assert(norm_elec.max() - norm_elec.min() > 0.0), (norm_elec.max(),norm_elec.min())
    denominator = norm_elec.max() - norm_elec.min() if norm_elec.max() - norm_elec.min() > 0.0 else 1.0       
    norm_elec = (norm_elec - norm_elec.min()) / denominator
    elec = np.stack((norm_elec, mask_elec))
    
    '''
    print(norm_elec.shape)
    print(mask_elec.shape)  
    print(elec.shape)  

    print(meta_elec_max.shape)     
    print(meta_elec_max.shape) 
    print(meta_elec_max.shape) 
    print(meta_elec_max.shape)      
    print(meta.shape)    
    '''
    
    return True, elec.astype(np.float32), meta.astype(np.float32)

"""# Processing dataset"""
def get_processed_dataset(filepath):
    df_raw = get_dataset(filepath)
    
    num = 1
    for uname, _ in df_raw.iterrows():
        if num % 200 == 0:
           print(num)
        num = num + 1
           
        #print(uname)
        data = df_raw.loc[uname].values
        flag = int(data[-1])
        elec = data[0:-1]
        #print('flag:', flag)
        #print(elec.shape)
        assert(elec.shape[0] % 7 ==0)
 
        IsEmpty, elec, meta = user_based_preprocessing(elec)
        if IsEmpty == False:
           print ("All NaN:", uname)
        else:
           save_path = 'datasets/electricity/abnormal/' if flag == 1 else 'datasets/electricity/normal/'
           np.savez(save_path + uname, elec=elec, meta=meta, flag=flag) 
        
def cross_validation_random_partition(path, sample_txt, cv_txt_prefix, cv_num = 5):
    all_samples = []
    for sample in open(path + sample_txt):  
         all_samples.append(sample)
    random.shuffle(all_samples)
    
    cv_unit_num = math.ceil(len(all_samples)/cv_num)
    for i in range(cv_num):
       start_id = cv_unit_num * i
       end_id = cv_unit_num * (i+1) if cv_unit_num * (i+1) <= len(all_samples) else len(all_samples)
       
       f=open(path + cv_txt_prefix + str(i) + '.txt', "w")
       for line in all_samples[start_id : end_id]:
           f.write(line)
       f.close()


def train_validation_test_random_partition(path, sample_txt, cv_txt_prefix, cv_num=5):
    all_samples = []
    for sample in open(path + sample_txt):
        all_samples.append(sample)
    random.shuffle(all_samples)

    cv_unit_num = math.ceil(len(all_samples) / cv_num)
    split = ["train", "val", "test"]
    start_id = {"train": 0,
                "val": cv_unit_num * 2,
                "test": cv_unit_num * 3}
    end_id = {"train": cv_unit_num * 2,
              "val": cv_unit_num * 3,
              "test": cv_unit_num * 5 if cv_unit_num * 5 <= len(all_samples) else len(all_samples)}
    for dataset in split:
        f = open(path + cv_txt_prefix + dataset + '.txt', "w")
        for line in all_samples[start_id[dataset]: end_id[dataset]]:
            f.write(line)
        f.close()

if __name__ == '__main__':

  filepath = 'datasets/electricity/data.csv'
  get_processed_dataset(filepath)

  # cross_validation_random_partition('datasets/electricity/', 'abnormal_all.txt', 'abnormal_cv_')
  # cross_validation_random_partition('datasets/electricity/', 'normal_all.txt', 'normal_cv_')
  # train_validation_test_random_partition('datasets/electricity/', 'abnormal_all.txt', 'abnormal_')
  # train_validation_test_random_partition('datasets/electricity/', 'normal_all.txt', 'normal_')