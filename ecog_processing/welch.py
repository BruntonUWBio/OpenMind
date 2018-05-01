from __future__ import print_function
# from __future__ import absolute_import
from __future__ import division

import glob
import json
import os
import re
import sys
import pickle
import csv
import mne
import numpy as np
import pandas as pd
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animation.OpenFaceScripts.scoring import AUScorer
from animation.OpenFaceScripts.scoring.EmotionPredictor import make_emotion_data
from scipy.io import loadmat
import joblib

from mne.io import read_raw_edf

tmin = -.2
tmax = .5
VIDEO_FPS = 30


def get_datetimes(raw, start, end):
    info = raw.info
    basetime_posix = info['meas_date']
    channel1 = raw.get_data(
        picks=1,
        start=start,
        stop=end,
        reject_by_annotation=None,
        return_times=True)
    datetimes = pd.to_datetime(channel1[1].ravel() + basetime_posix, unit='s')

    return datetimes


def get_events(filename,
               au_emote_dict,
               classifier_loc,
               real_time_file_loc: str,
               emotion='Happy') -> tuple:
    events = []
    classifier = pickle.load(
        open(
            os.path.join(
                classifier_loc,
                '{0}_trained_RandomForest_with_pose.pkl'.format(emotion)),
            'rb'))
    aus_list = AUScorer.TrainList
    times = []
    predicted_arr = []
    patient_session = os.path.basename(filename).replace('.edf', '')
    # op_folder = '/data2/OpenFaceTests'
    patient_folders = (x for x in au_emote_dict if patient_session in x)
    real_time_file = os.path.join(real_time_file_loc, patient_session + '.csv')

    if not os.path.exists(real_time_file):
        return None, None, None
    real_time_dict = {}

    with open(real_time_file) as real_time_info:
        real_time_reader = csv.reader(real_time_info)

        for row in real_time_reader:
            vid_name = row[0]
            year, month, day, hour, minute, second, microseconds = map(
                int, row[1:])
            real_world_time = datetime.datetime(year, month, day, hour, minute,
                                                second, microseconds)
            real_time_dict[vid_name.replace('.avi', '')] = real_world_time

    for patient_folder in patient_folders:
        presence_dict = au_emote_dict[patient_folder]

        if presence_dict and any(presence_dict.values()):
            curr_patient = patient_folder.replace('_cropped', '')

            if curr_patient in real_time_dict:
                curr_patient_start_time = real_time_dict[
                    patient_folder.replace('_cropped', '')]

                for frame in presence_dict:
                    elapsed_seconds = int(frame) / VIDEO_FPS
                    frame_time = curr_patient_start_time + \
                        datetime.timedelta(seconds=elapsed_seconds)

                    times.append(frame_time)

                    if presence_dict[frame]:
                        aus = presence_dict[frame][0]
                        au_data = ([float(aus[str(x)]) for x in aus_list])
                        predicted = classifier.predict_proba(
                            np.array(au_data).reshape(1, -1))[0]
                        predicted_arr.append(predicted)
                        classified = classifier.predict(
                            np.array(au_data).reshape(1, -1))

                        if classified[0] == 1:
                            events.append([frame_time, 0, 1])
                    else:
                        predicted_arr.append(np.array([np.NaN, np.NaN]))

    corr = [x[1] for x in predicted_arr]

    return events, times, corr
    # # presence_dict = json.load(open(os.path.join(op_folder, patient_folder, 'all_dict.txt')))
    # if presence_dict:
    #     frames = presence_dict.keys()


def load_montage():
    mat = loadmat('/home/gvelchuru/ecb43e/ecb43e_Montage.mat')
    array = np.array(mat['Montage'][0])

    return array


def get_ecg_arr(epochs: mne.Epochs) -> np.ndarray:
    epochs.plot(mne.pick_types(epochs.info, meg=False, ecg=True))
    evoked = epochs.average(mne.pick_types(epochs.info, meg=False, ecg=True))
    evoked.plot()

    return np.zeros(1)


if __name__ == '__main__':

    edf_dir = sys.argv[sys.argv.index('-e') + 1]

    # filenames = glob.iglob("/data1/**/*.edf", recursive=True)
    # filenames = ['/data1/edf/a1d36553/a1d36553_4.edf']
    au_emote_dict = json.load(open('/data2/OpenFaceTests/au_emotes.txt'))

    for filename in filenames:
        # try:
        print(filename)

        raw = read_raw_edf(filename, preload=False)
        start = 200000
        end = 400000
        # datetimes = get_datetimes(raw, start, end)
        mapping = {
            ch_name: 'ecog'
            for ch_name in raw.ch_names if 'GRID' in ch_name
        }
        mapping.update(
            {ch_name: 'ecg'
             for ch_name in raw.ch_names if 'ECG' in ch_name})
        mapping.update({
            ch_name: 'eeg'
            for ch_name in raw.ch_names if ch_name not in mapping
        })
        raw.set_channel_types(mapping)

        # raw.set_montage(mon)
        picks = mne.pick_types(raw.info, ecog=True)
        # picks = picks[10:30]
        # data = raw.get_data(picks, start, end)
        events, corr_arr = get_events(filename, au_emote_dict)

        np.save('corr_arr.npy', corr_arr)

        if len(events) > 0:
            # raw.save('test.raw.fif')
            epochs = mne.Epochs(raw, events, preload=True)
            epochs.pick_types(epochs.info, ecog=True, ecg=True, eeg=False)
            # evoked = epochs.average(picks=picks)
            # mat = loadmat('/home/gvelchuru/ecb43e/trodes.mat')
            #
            # elec = mat['Grid']
            # ch_names = list(map(str, picks[:len(elec)]))
            # evoked = evoked.pick_channels(evoked.ch_names[:len(elec)])
            # # temporary
            # evoked.rename_channels({ch_name: i for ch_name, i in zip(evoked.ch_names, ch_names)})
            #
            # dig_ch_pos = dict(zip(ch_names, elec))
            # mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos, point_names=ch_names)

            mat = loadmat('/home/gvelchuru/ecb43e/trodes.mat')

            # np.save('ecg_arr.npy', get_ecg_arr(
            epochs.save('test-epo.fif')

            # evoked.set_montage(mon)
            # evoked.save('test-ave.fif')
            # except RuntimeError:
            #     print('error \t' + filename)
            #     continue
