# from __future__ import absolute_import, division

import glob
import json
import os
import re
import sys
import pickle

import mne
import numpy as np
import pandas as pd
from collections import defaultdict

from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.scoring.EmotionPredictor import make_emotion_data
from scipy.io import loadmat
import joblib

sys.path.append('/home/gvelchuru')
from mne.io import read_raw_edf

tmin = -.2
tmax = .5

# fig, ax = plt.subplots(1)
# raw = mne.io.read_raw_edf('/data1/edf/a1d36553/a1d36553_8.edf', preload=False)

# info = raw.info
# basetime_posix = info['meas_date']

# chanel1 = raw.get_data(picks=1, start=0, stop = 400000, reject_by_annotation=None, return_times= True)
#
# picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
# picks = picks[10:25]
#
# fmin, fmax = .1, 160
# n_fft = 4096
#
# tmin = raw.times[0]
# tmax = raw.times[raw.n_times - 1]
#
# psd, freqs = psd_welch(raw, fmin, fmax, n_fft=n_fft, tmin=2000, tmax=10000, n_jobs=-1, picks=picks)
# cmap = 'RdBu_r'
# freq_mask = freqs < 10
# freqs = freqs[freq_mask]
# log_psd = 10 * np.log10(psd)
#
# im = ax.imshow(log_psd[:, freq_mask].T, aspect='auto', origin='lower', cmap=cmap)
#
# ax.set_yticks(np.arange(0, len(freqs), 10))
# ax.set_yticklabels(freqs[::10].round(1))
# ax.set_ylabel('Frequency (Hz)')
# ax.set_xticks(np.arange(0, len(picks)))
# ax.set_xticklabels(picks)
# ax.set_xlabel('EEG channel index')
# im.set_clim()
# plt.title('continous power spectrum from {0} to {1}'.format(tmin, tmax))
# plt.show()


# mat_array = mat['Montage']
# open('montage.txt', 'w').write(str(mat_array))
# file_mat_array = np.fromfile('montage.txt')


# ch_names = list(map(str, range(mat['Grid'].shape[0])))
# elec = mat['Grid']


# mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
# ch_names = mat['ch_names'].tolist()
# elec = mat['elec']

# dig_ch_pos = dict(zip(ch_names, elec))
# mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos, point_names=ch_names)
# print('Created %s channel positions' % len(ch_names))
# info = mne.create_info(ch_names, 1000., 'ecog', montage=mon, verbose=3)
# subjects_dir = mne.datasets.sample.data_path() + '/subjects'
#
# fig = plot_alignment(info, subject='sample', meg=False, subjects_dir=subjects_dir, surfaces=['pial'], verbose=3)
# mlab.savefig('sample_alignment.png', figure=fig)
# # mlab.view(200, 70)

# mon.transform_to_head()
# mon.save('montage.fif')

# raw = read_raw_edf('/home/gvelchuru/a1d36553_8.edf')


# print(raw[0:100])
# events = mne.find_events(raw)
# raw.save('test.raw.fif')


# evoked = mne.Evoked(None, stim_cha)
# evoked.set_montage(mon)


def get_datetimes(raw, start, end):
    info = raw.info
    basetime_posix = info['meas_date']
    channel1 = raw.get_data(picks=1, start=start, stop=end, reject_by_annotation=None, return_times=True)
    datetimes = pd.to_datetime(channel1[1].ravel() + basetime_posix, unit='s')
    return datetimes


def get_events(filename, au_emote_dict, emotion='Happy'):
    events = []
    classifier = pickle.load(open('/data2/OpenFaceTests/{0}_trained_RandomForest_with_pose.pkl'.format(emotion), 'rb'))
    aus_list = AUScorer.TrainList
    predicted_arr = []
    patient_session = os.path.basename(filename).replace('.edf', '')
    # op_folder = '/data2/OpenFaceTests'
    patient_folders = (x for x in au_emote_dict if patient_session in x)
    for patient_folder in patient_folders:
        presence_dict = au_emote_dict[patient_folder]
        if presence_dict and any(presence_dict.values()):
            nums = re.findall(r'\d+', patient_folder)
            session = int(nums[len(nums) - 1])
            starting_time = int(session * 120 * 1000)  # convert to sampling rate
            for frame in presence_dict:
                if presence_dict[frame] and presence_dict[frame][1] == emotion:
                    events.append(([int(starting_time + int(frame) * (1000 / 30)), 0, 1]))
                    past_frame = int(int(frame) + tmin * 30)
                    future_frame = int(int(frame) + tmax * 30)
                    for frame_to_add in map(str, range(past_frame, future_frame + 1)):
                        if frame_to_add in presence_dict and presence_dict[frame_to_add]:
                            aus = presence_dict[frame_to_add][0]
                            au_data = ([float(aus[str(x)]) for x in aus_list])
                            predicted = classifier.predict_proba(np.array(au_data).reshape(1, -1))[0]
                            predicted_arr.append(predicted)
                        else:
                            predicted_arr.append(np.array([0, 0]))
    # au_data, _ = make_emotion_data(emotion, evaluate_dict, False)
    # predicted_emotes = classifier.predict(au_data)
    times = [x for x in range(0, len(predicted_arr), 10)]
    corr = [predicted_arr[a][1] for a in times]
    return np.array(events, dtype=np.int), np.array((times, corr))
    # presence_dict = json.load(open(os.path.join(op_folder, patient_folder, 'all_dict.txt')))
    # if presence_dict:
    #     frames = presence_dict.keys()


def load_montage():
    mat = loadmat('/home/gvelchuru/ecb43e/ecb43e_Montage.mat')
    array = np.array(mat['Montage'][0])

if __name__ == '__main__':
    # filenames = glob.iglob("/data1/**/*.edf", recursive=True)
    # filenames = ['/data1/edf/a1d36553/a1d36553_4.edf']
    filenames = ['cb46fd46_7.edf']
    au_emote_dict = json.load(open('/data2/OpenFaceTests/au_emotes.txt'))
    for filename in filenames:
        # try:
        print(filename)

        raw = read_raw_edf(filename, preload=False)
        start = 200000
        end = 400000
        # datetimes = get_datetimes(raw, start, end)
        mapping = {ch_name: 'ecog' for ch_name in raw.ch_names}
        open('ch_names.txt', 'w').write(str(raw.ch_names))
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
            # load_montage()

            elec = mat['Grid']
            ch_names = list(map(str, picks[:len(elec)]))
            epochs = epochs.pick_channels(epochs.ch_names[:len(elec)])

            # temporary
            epochs.rename_channels({ch_name: i for ch_name, i in zip(epochs.ch_names, ch_names)})

            dig_ch_pos = dict(zip(ch_names, elec))
            mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos, point_names=ch_names)

            epochs.save('test-epo.fif')



            # evoked.set_montage(mon)
            # evoked.save('test-ave.fif')
            # except RuntimeError:
            #     print('error \t' + filename)
            #     continue

