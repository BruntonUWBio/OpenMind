#################################################################################################
#
# combine_off.py is a script to combine Open Face Features from different videos to one csv file
#
# Usage: python combine_off.py /data2/OpenFaceTests/ /data2/OpenFaceFeatures/joined_features.csv
# 
# The first argument corresponds to the pattern we want to match in the action units folder.
# For example, if we specify /data2/OpenFaceTests/a86a4375 we will store the features only for 
# one patient.
# 
#################################################################################################

import sys
import json
import os

import progressbar
from numpy import isclose

import pandas as pd
from glob import glob

name = sys.argv[1]
folder_list = glob(name + '*_cropped')
df = pd.DataFrame()

# opening the output file
if os.path.exists(sys.argv[2]):
    os.remove(sys.argv[2])

f = open(sys.argv[2], 'a')
H = True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

bar = progressbar.ProgressBar(max_value=len(folder_list))
curr_end = 0
# get column  names
df = pd.read_csv('/data2/OpenFaceFeatures/features_a86a4375.csv',nrows=0)
print((df.columns))
print(len(set(df.columns)))

new_columns = []
for field in df.columns:
    if 'AU' in field:
        new_columns.append(field[3:-2])
    else:
        new_columns.append(field.strip())

new_columns = list(set(new_columns))  
df_empty = pd.DataFrame(columns=new_columns)


# create a joint collection of the column names without writing anything

col_names = []
for folder_num, folder in enumerate(folder_list):
    
    if 'all_dict.txt' in os.listdir(folder):
        # multi_runs_dict = json.load(open(os.path.join(folder, 'all_dict.txt')))
        data = pd.read_json(os.path.join(folder,'all_dict.txt')).T
        col_names = col_names + list(data.columns)

print("all columns")
print(len(set(col_names)))   
print(col_names)
# add the extra columns
col_names = col_names + ['patient','session','vid','frame']

df_empty = pd.DataFrame(columns = col_names)

for folder_num, folder in enumerate(folder_list):
    df_empty = pd.DataFrame(columns=list(set(col_names)))
    temp = pd.read_csv(folder + '/au.txt', index_col=0)
    dir_names = folder.split('/')
    patient, session, vid, cropped = dir_names[-1].split('_')
    temp['patient'] = patient
    temp['session'] = session
    temp['vid'] = vid
    temp.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)
    if 'all_dict.txt' in os.listdir(folder):
        # multi_runs_dict = json.load(open(os.path.join(folder, 'all_dict.txt')))
        data = pd.read_json(os.path.join(folder,'all_dict.txt')).T.reset_index()
        # add a check whether there is something in all_dict
        data['patient'] = patient
        data['session'] = session
        data['vid'] = vid
        print(data.columns)
        data.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)
        data.rename(columns={'index':'frame'}, inplace=True)
        print(data.columns)
        df_empty = df_empty.append(data,ignore_index=True)
    
    df_empty.to_csv(f, header=H)
    H = False
    curr_end = len(temp)
    bar.update(folder_num)
# df.to_csv(sys.argv[2])
print(df_empty.columns)
