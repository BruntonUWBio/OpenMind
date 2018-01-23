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
for folder_num, folder in enumerate(folder_list):
    # if 'all_dict.txt' in os.listdir(folder):
    #     temp_au = pd.read_csv(folder + '/au.txt', index_col = 0)
    #     temp = pd.read_json(os.path.join(folder, 'all_dict.txt'))
    #     temp.join(temp_au)
    #     pass
    # else:
    temp = pd.read_csv(folder + '/au.txt', index_col=0)
    dir_names = folder.split('/')
    patient, session, vid, cropped = dir_names[-1].split('_')
    temp['patient'] = patient
    temp['session'] = session
    temp['vid'] = vid
    temp.rename(columns={u'Unnamed: 0': u'frame3'}, inplace=True)
    if 'all_dict.txt' in os.listdir(folder):
        multi_runs_dict = json.load(open(os.path.join(folder, 'all_dict.txt')))
        for frame, frame_dict in multi_runs_dict.items():
            num_frame = int(frame)
            row_index = curr_end + num_frame
            temp.at[row_index, 'success'] = 1
            for field, val in frame_dict.items():
                if is_number(field):
                    field = 'AU' + field
                    temp.at[row_index, field + '_c'] = 1
                    temp.at[row_index, field + '_r'] = val
                else:
                    temp.at[row_index, field] = val
    # df = df.append(temp)
    temp.to_csv(f, header=H)
    H = False
    curr_end = len(temp)
    bar.update(folder_num)
# df.to_csv(sys.argv[2])
