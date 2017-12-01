"""
align_vid_ecog.py converts vid_real_time files to a more appropriate format for indexing
final table is of this form
------------------------------
|patient|session|vid|datetime|
|  ...  |  ...  |...|  ...   |


Usage:  python align_vid_ecog.py /data1/sharedata/vid_real_time/a86a4375_2 output.csv


"""


import pandas as pd
import sys

# reading the file for this session
session_name = sys.argv[1]
vid_times  = pd.read_csv(session_name+'.csv', header = None)
vid_times.columns = ['vid_name','year','month','day','h','m','s','us']


# convering the video file to  index
foo = lambda x: pd.Series([i for i in x.split('_')])
new = vid_times['vid_name'].apply(lambda x: x[:-4]).apply(foo)
new.columns = ['patient','session','vid']

# converting to datetime
new['datetime'] = pd.to_datetime(vid_times[['year','month','day','h','m','s','us']], unit='us')

print(new.head())

output = sys.argv[2]
new.to_csv(output,index = None)

