############################################################################################################
#
# join_realtime adds a column of realtimes to the features. 
# 
# Usage: python join_realtime.py joined_features.csv patient_session datetime_folder 
# Example: python join_realtime.py /data2/OpenFaceFeatures/joined.csv ../temp_data/
# 
###########################################################################################################

import sys
session_name =  sys.argv[2]

import dask.dataframe as dd
import pandas as pd

# reading the large file using dask
filename = sys.argv[1]
features = dd.read_csv(filename, assume_missing = True)
vid_times = pd.read_csv(sys.argv[3]+'datetime_'+session_name+'.csv')
patient,session = session_name.split("_")
# features_patient = features[(features.patient==patient) and (features.session==float(session))].compute()
# extract features for a specific patient
features_patient = features[(features.patient==patient)].compute()
print(features_patient.shape)

# merging the two tables
merged = pd.merge(features_patient[features_patient['session'] == float(session)], vid_times,how = 'left' )
merged.rename(columns = {' timestamp':'timestamp'},inplace = True)
merged['realtime'] = pd.to_datetime(merged.datetime)+pd.to_timedelta(merged.timestamp)
merged.to_csv('/data2/OpenFaceFeatures/features_'+session_name+'_realtime.csv',index = False)
