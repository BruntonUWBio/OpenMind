############################################################################################################
#
# join_realtime adds a column of realtimes to the features. 
# 
# Usage: python join_realtime.py patient_session features_patient_session.csv
#
# 
###########################################################################################################

import sys
session_name =  sys.argv[1]

import dask.dataframe as dd
import pandas as pd

# reading the large file using dask
filename = '/data2/OpenFaceFeatures/joined.csv'
features = dd.read_csv(filename, assume_missing = True)
vid_times = pd.read_csv('~/ecogAnalysis/data_preprocessing/datetime_'+session_name+'.csv')
patient,session = session_name.split("_")
# features_patient = features[(features.patient==patient) and (features.session==float(session))].compute()
# extract features for a specific patient
features_patient = features[(features.patient==patient)].compute()
print(features_patient.shape)

# merging the two tables
merged = pd.merge(features_patient[features_patient['session'] == float(session)], vid_times,how = 'left' )
merged.rename(columns = {' timestamp':'timestamp'},inplace = True)
merged['realtime'] = pd.to_datetime(merged.datetime)+pd.to_timedelta(merged.timestamp)
merged.to_csv('/data2/OpenFaceFeatures/features_'+session_name+'.csv',index = False)
