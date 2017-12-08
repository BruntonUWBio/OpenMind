import sys
session_name =  sys.argv[1]

import dask.dataframe as dd
import pandas as pd

filename = '/data2/OpenFaceFeatures/joined.csv'
features = dd.read_csv(filename, assume_missing = True)
vid_times = pd.read_csv('~/ecogAnalysis/data_preprocessing/datetime_'+session_name+'.csv')
patient,session = session_name.split("_")
# features_patient = features[(features.patient==patient) and (features.session==float(session))].compute()
features_patient = features[(features.patient==patient)].compute()
print(features_patient.shape)
merged = pd.merge(features_patient[features_patient['session'] == float(session)], vid_times,how = 'left' )
merged.rename(columns = {' timestamp':'timestamp'},inplace = True)
merged['realtime'] = pd.to_datetime(merged.datetime)+pd.to_timedelta(merged.timestamp)
merged.to_csv('/data2/OpenFaceFeatures/features_'+session_name+'.csv',index = False)
