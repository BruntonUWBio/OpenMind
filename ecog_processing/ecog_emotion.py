import numpy as np
import mne
import json
import random
import multiprocessing
import sys
from glob import glob
from mne.time_frequency import psd_welch
from mne.io import read_raw_edf
from welch import get_events
from pathos.multiprocessing import ProcessingPool as Pool
import functools


def find_filename_data(au_emote_dict, one_data, zero_data, filename):
    print(filename)
    raw = read_raw_edf(filename, preload=False)
    # start = 200000
    # end = 400000
    # datetimes = get_datetimes(raw, start, end)
    mapping = {ch_name: 'ecog' for ch_name in raw.ch_names
               if 'GRID' in ch_name}
    mapping.update(
        {ch_name: 'ecg' for ch_name in raw.ch_names if 'ECG' in ch_name})
    mapping.update(
        {ch_name: 'eeg' for ch_name in raw.ch_names
         if ch_name not in mapping})
    raw.set_channel_types(mapping)
    # raw.set_montage(mon)
    # picks = picks[10:30]
    # data = raw.get_data(picks, start, end)
    events, corr_arr = get_events(filename, au_emote_dict)
    print('making event_time_dict')
    times = corr_arr[0]
    timePredics = corr_arr[1]
    predicDic = {time: predic for time, predic in zip(times, timePredics)}
    eventTimes = set(x[0] for x in events)
    picks = mne.pick_types(raw.info, ecog=True)
    all_times = len(raw)
    num_divisions = all_times / 1000
    range_times = np.arange(all_times)
    range_times = np.array_split(range_times, num_divisions)
    for time_arr in range_times:
        time_start = raw.time_as_index(time_arr[0])
        time_end = raw.time_as_index(time_arr[len(time_arr) - 1])
        data, times = raw[picks, time_start:time_end]
        begin_time = times[0]
        end_time = times[len(times) - 1]
        has_event = False
        annotated = False
        # all_nans = True
        for time in np.arange(begin_time, end_time):
            if time in predicDic and not np.isnan(predicDic[time]):
                annotated = True
                break
        if annotated:
            for time in np.arange(begin_time, end_time):
                if time in eventTimes:
                    has_event = True
                    break
            # if not all_nans:
            psd = psd_welch(data, 32, 100)
            if has_event:
                one_data.append(psd)
            else:
                zero_data.append(psd)


if __name__ == '__main__':
    edf_dir = sys.argv[sys.argv.index('-e') + 1]
    my_comp = sys.argv[sys.argv.index('-c') + 1]
    au_emote_dict = json.load(open(sys.argv[sys.argv.index('-au') + 1]))
    m = multiprocessing.Manager()
    zero_data = m.list()
    one_data = m.list()
    filenames = glob(edf_dir)
    f = functools.partial(find_filename_data, au_emote_dict,
                          one_data, zero_data)
    Pool().map(f, filenames)
    # find_filename_data(au_emote_dict, one_data, zero_data, filenames[0])

    random.shuffle(zero_data)
    zero_data = zero_data[:len(one_data)]
    all_data = []
    all_labels = []
    for datum in zero_data:
        all_data.append(datum)
        all_labels.append(0)
    for datum in one_data:
        all_data.append(datum)
        all_labels.append(1)
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    np.save('classifier_data/all_{0}_data.npy'.format(my_comp), all_data)
    np.save('classifier_data/all_{0}_labels.npy'.format(my_comp), all_labels)
