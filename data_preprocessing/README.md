The Open Face Features obtained by running Open Face are stored in cylon@/data2/OpenFaceTests. The features for each video are stored in a folder patient_session_vid_cropped file. We can combine them into one big .csv file.

Step 0: combine features into one .csv file

```
   python combine_off.py /data2/OpenFaceTests/patient_session /data2/OpenFaceFeatures/features_patient_session.csv
```
(Here we just show how to process the data for one session, similarly all features can be combined, or features for one patient)

To obtain real times in the open face features, we need to add the beginning real time of each video to the timestamp for each timestamp in the features table.

This can be obtained by merging the features file with the video times file.

Step 1: convert the video times file in a datetime format:

```
 python convert_datetime.py patient_session.csv datetime_patient_session.csv 
```

Step 2: merge the features with their realtimes

```
 python join_realtime.py patient_session features_patient_session.csv
```

In the end `features_patient_session.csv` has a column `realtime`.


Warnings:

* Some column name come up with a space in front of the name!
* When the final files are read in pandas not all datetime columns are read as datetime, but they can be converted using the `pd.to_datetime` command.

