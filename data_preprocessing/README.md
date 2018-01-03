The Open Face Features obtained by running Open Face are stored in cylon@/data2/OpenFaceTests. The features for each video are stored in a folder patient_session_vid_cropped file. We can combine them into one big .csv file.

Step 0: combine features into one .csv file

one session:

```
   python combine_off.py /data2/OpenFaceTests/patient_session /data2/OpenFaceFeatures/features_patient_session.csv
```

all:

```
   python combine_off.py /data2/OpenFaceTests/ /data2/OpenFaceFeatures/joined.csv
```

To obtain real times in the open face features, we need to add the beginning real time of each video to the timestamp for each timestamp in the features table.

This can be obtained by merging the features file with the video times file.

Step 1: convert the video times file in a datetime format:

```
 python convert_datetime_per_session.py /data1/sharedata/vid_real_time/fcb01f7a_2 ../temp_data/datetime_fcb01f7a_2.csv
```

(If you want to convert all files simultaneously run:

```
 python convert_datetime_all.py /sharedata/ ../temp_data
```

That will generate a datetime_patient.csv file for each patient.)

Step 2: merge the features with their realtimes


```
  python join_realtime.py /data2/OpenFaceFeatures/joined.csv fcb01f7a_2 ../temp_data/
```

In the end `features_patient_session_realtime.csv` in the OpenFaceFeatures folder has a column `realtime`.


Warnings:

* Some column name come up with a space in front of the name!
* When the final files are read in pandas not all datetime columns are read as datetime, but they can be converted using the `pd.to_datetime` command.

