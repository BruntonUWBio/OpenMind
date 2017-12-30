import pandas as pd
import sys
import os
import argparse
from argparse import RawTextHelpFormatter


"""convert_datetime.py converts vid_real_time files to a more appropriate format for indexing
final table is of this form
------------------------------
|patient|session|vid|datetime|
|  ...  |  ...  |...|  ...   |
Usage:  python convert_datetime.py /data1/sharedata/vid_real_time/a86a4375_2 output.csv"""

# parser = argparse.ArgumentParser(description="")
# formatter_class=RawTextHelpFormatter)
# parser = argparse.ArgumentParser()
# parser.add_argument(help="help")

# patient names
from glob import glob
n = len(sys.argv[1])
filenames = glob(sys.argv[1]+'*.csv')

# get unique patient names
patient_ids = [filename.split('_')[0] for filename in filenames]
unique_patient_ids = list(set(patient_ids))
print(unique_patient_ids)

# for patient in unique_patient_ids:  

# do not need to use unique  
for filename in filenames[0:]:
    patient,session = filename[n:-4].split('_')
    print(patient)
    print(session)    
    vid_times  = pd.read_csv(filename, header = None)
    print(vid_times.head())
    ncols = vid_times.shape[1]
    colnames = ['vid_name','year','month','day','h','m','s','us']
    vid_times.columns = colnames[:ncols]
    # convering the video file to  index
    f1 = lambda x: pd.Series([i for i in x.split('_')])
    f2 = lambda x: x.split('/')[-1]
    f3 = lambda x: os.path.normpath(x).split('\\')[-1]
    print(vid_times['vid_name'])
    new = vid_times['vid_name'].apply(lambda x: x[:-4]).apply(f3).apply(f2).apply(f1)
    print(new.columns)
    new.columns = ['patient','session','vid']

    # converting to datetime
    new['datetime'] = pd.to_datetime(vid_times[colnames[1:ncols]], unit=colnames[ncols-1])
    print(new.head())
    output = sys.argv[2]+patient+'.csv'
    new.to_csv(output,index = None, mode='a')

