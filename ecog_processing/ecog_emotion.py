import numpy as np
import mne
import json
import random
import sys
import os
import datetime
import multiprocessing
import sys
import os
from glob import glob
from mne.time_frequency import psd_welch
from mne.io import read_raw_edf
from welch import get_events
from pathos.multiprocessing import ProcessingPool as Pool
import functools
from tqdm import tqdm


def find_filename_data(au_emote_dict_loc, one_data, zero_data, classifier_loc,
                       filename):
    print(filename)
    au_emote_dict = json.load(open(au_emote_dict_loc))
    raw = read_raw_edf(filename, preload=False)
    # start = 200000
    # end = 400000
    # datetimes = get_datetimes(raw, start, end)
    mapping = {
        ch_name: 'ecog'
        for ch_name in raw.ch_names if 'GRID' in ch_name
    }
    mapping.update(
        {ch_name: 'ecg'
         for ch_name in raw.ch_names if 'ECG' in ch_name})
    mapping.update(
        {ch_name: 'eeg'
         for ch_name in raw.ch_names if ch_name not in mapping})

    if 'ecog' not in mapping.value():
        return

    raw.set_channel_types(mapping)

    # raw.set_montage(mon)
    # picks = picks[10:30]
    # data = raw.get_data(picks, start, end)
    events, times, corr = get_events(filename, au_emote_dict, classifier_loc)
    predicDic = {time: predic for time, predic in zip(times, corr)}
    eventTimes = set(x[0] for x in events)
    picks = mne.pick_types(raw.info, ecog=True)

    if len(picks) > 0:
        all_times = len(raw)
        ecog_start_time = sorted(times)[0]
        num_divisions = all_times / 1000
        range_times = np.arange(all_times)
        range_times = np.array_split(range_times, num_divisions)

        for time_arr in range_times:
            ecog_time_arr_start = raw.time_as_index(time_arr[0])[0]
            ecog_time_arr_end = raw.time_as_index(
                time_arr[len(time_arr) - 1])[0]
            data, ecog_times = raw[picks, ecog_time_arr_start:
                                   ecog_time_arr_end]
            real_window_start = ecog_start_time + \
                datetime.timedelta(seconds=time_arr[0]/1000)
            real_window_end = ecog_start_time + \
                datetime.timedelta(seconds=time_arr[len(time_arr) - 1]/1000)
            has_event = False

            for event_time in eventTimes:
                if real_window_start <= event_time <= real_window_end:
                    has_event = True

                    break
            psd = psd_welch(data, 32, 100)

            if has_event:
                one_data.append(psd)
            else:
                zero_data.append(psd)
    else:
        print('no picks')


if __name__ == '__main__':
    edf_dir = sys.argv[sys.argv.index('-e') + 1]
    my_comp = sys.argv[sys.argv.index('-c') + 1]
    au_emote_dict_loc = (sys.argv[sys.argv.index('-au') + 1])
    classifier_loc = sys.argv[sys.argv.index('-cl') + 1]
    m = multiprocessing.Manager()
    zero_data = m.list()
    one_data = m.list()
    filenames = glob(os.path.join(edf_dir, '**/*.edf'), recursive=True)
    f = functools.partial(find_filename_data, au_emote_dict_loc, one_data,
                          zero_data, classifier_loc)
    with tqdm(total=len(filenames)) as pbar:
        p = Pool()

        for iteration, _ in enumerate(p.uimap(f, filenames)):
            pbar.update()
        p.close()
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
