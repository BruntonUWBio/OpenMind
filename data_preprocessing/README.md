To obtain real times in the open face features, we need to add the beginning real time of each video to the timestamp for each timestamp in the features table.

This can be obtained by merging the features file with the video names file.

First run python align_ecog_vis.py session_name.csv output.csv to arrange the indeces. 

```
features = pd.read_csv('~/features.csv')
vid_times = pd.read_csv('~/ecogAnalysis/data_preprocessing/output.csv')
merged = pd.merge(joined, vid_times,how = 'left' )
```

In the end add the timestamp column to the datatime column:
merged['real_time'] = merged['timestamp'] + merged['datetime']

Make sure they are read in proper data format.

