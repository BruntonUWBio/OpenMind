#################################################################################################
#
# combine_off.py is a script to combine Open Face Features from different videos to one csv file
#
# Usage: python combine_off.py /data2/OpenFaceTests/ /data2/OpenFaceFeatures/joined_features.csv
# 
# The first argument corresponds to the pattern we want to match in the action units folder.
# For example, if we specify /data2/OpenFaceTests/a86a4375 we will store the features only for 
# patient.
# The second argument corresponds to the name of the output file.
# 
#################################################################################################

import sys
import pandas as pd
from glob import glob
name = sys.argv[1]
folder_list = glob(name+'*_cropped')
df = pd.DataFrame()

# opening the output file
f = open(sys.argv[2],'a')
H = True
for folder in folder_list:
    temp = pd.read_csv(folder+'/au.txt',index_col = 0)
    dir_names = folder.split('/')   
    patient,session,vid,cropped= dir_names[-1].split('_')   
    temp['patient'] = patient
    temp['session'] = session
    temp['vid'] = vid
    temp.rename(columns = {u'Unnamed: 0': u'frame3'},inplace = True) 
    # df = df.append(temp)
    temp.to_csv(f, header = H)
    H = False
# df.to_csv(sys.argv[2])
