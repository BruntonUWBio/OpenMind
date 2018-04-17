import argparse
import json
import random
import sys
import os
import datetime
import multiprocessing
import sys
import os
import numpy as np
import mne
from glob import glob
from mne.time_frequency import psd_welch
from scipy.signal import welch
from mne.io import read_raw_edf
from welch import get_events
from pathos.multiprocessing import ProcessingPool as Pool
import functools
from tqdm import tqdm
from typing import List, Dict

def get_window_data(raw: mne.io.Raw, times: list, picks,
                                        eventTimes: set) -> tuple:
        ECOG_SAMPLING_FREQUENCY = 1000  # ECoG samples at a rate of 1000 Hz
            EVENT_DELTA_SECONDS = 1
                all_times = len(raw)
                    ecog_start_time = sorted(times)[0]
                        one_data = []
                            zero_data = []
                                curr_window = []  # type: List[int]
                                    emotion_on = False
                                        range_times = np.arange(all_times)

                                            for time in range_times:
                                                        real_pos = ecog_start_time + \
                                                                        datetime.timedelta(seconds=time / ECOG_SAMPLING_FREQUENCY)
                                                                emote_window_end = real_pos + \
                                                                                datetime.timedelta(seconds=time / ECOG_SAMPLING_FREQUENCY)
                                                                        has_event = False

                                                                                for event_time in eventTimes:
                                                                                                if real_pos <= event_time <= emote_window_end:
                                                                                                                    has_event = True

                                                                                                                                    break

                                                                                                                                        if not curr_window:
                                                                                                                                                        curr_window = [time]
                                                                                                                                                                    emotion_on = has_event

                                                                                                                                                                            elif emotion_on != has_event:
                                                                                                                                                                                            curr_window.append(time)
                                                                                                                                                                                                        ecog_time_arr_start = raw.time_as_index(curr_window[0])[0]
                                                                                                                                                                                                                    ecog_time_arr_end = raw.time_as_index(curr_window[1])[0]
                                                                                                                                                                                                                                data, ecog_times = raw[picks, ecog_time_arr_start:
                                                                                                                                                                                                                                                                                          ecog_time_arr_end]
                                                                                                                                                                                                                                            psd = psd_welch(data, 32, 100)

                                                                                                                                                                                                                                                        if emotion_on and not has_event:
                                                                                                                                                                                                                                                                            one_data.append(psd)
                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                            zero_data.append(psd)
                                                                                                                                                                                                                                                                                                                        curr_window = [time]

                                                                                                                                                                                                                                                                                                                            return one_data, zero_data


                                                                                                                                                                                                                                                                                                                        def find_filename_data(au_emote_dict_loc, one_data: list, zero_data: list,
                                                                                                                                                                                                                                                                                                                                                                      classifier_loc, filename, real_time_file: str):
                                                                                                                                                                                                                                                                                                                                print(filename)
                                                                                                                                                                                                                                                                                                                                    au_emote_dict = json.load(open(au_emote_dict_loc))
                                                                                                                                                                                                                                                                                                                                        raw = read_raw_edf(filename, preload=False)
                                                                                                                                                                                                                                                                                                                                            # start
                                                                                                                                                                                                                                                                                                                                            # =
                                                                                                                                                                                                                                                                                                                                            # 200000
                                                                                                                                                                                                                                                                                                                                                # end
                                                                                                                                                                                                                                                                                                                                                # =
                                                                                                                                                                                                                                                                                                                                                # 400000
                                                                                                                                                                                                                                                                                                                                                    # datetimes
                                                                                                                                                                                                                                                                                                                                                    # =
                                                                                                                                                                                                                                                                                                                                                    # get_datetimes(raw,
                                                                                                                                                                                                                                                                                                                                                    # start,
                                                                                                                                                                                                                                                                                                                                                    # end)
                                                                                                                                                                                                                                                                                                                                                    mapping = {
                                                                                                                                                                                                                                                                                                                                                                ch_name: 'ecog'
                                                                                                                                                                                                                                                                                                                                                                for ch_name in raw.ch_names if 'GRID' in ch_name

                                                                                                                                                                                                                                                                                                                                                    }  # type: Dict[str, str]
                                                                                                                                                                                                                                                                                                                                                    mapping.update(
                                                                                                                                                                                                                                                                                                                                                                {ch_name: 'ecg'
                                                                                                                                                                                                                                                                                                                                                                          for ch_name in raw.ch_names if 'ECG' in ch_name}
                                                                                                                                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                                                                                                                                    mapping.update(
                                                                                                                                                                                                                                                                                                                                                                {ch_name: 'eeg'
                                                                                                                                                                                                                                                                                                                                                                          for ch_name in raw.ch_names if ch_name not in mapping}
                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                        if 'ecog' not in mapping.values():
                                                                                                                                                                                                                                                                                                                                                                    return

                                                                                                                                                                                                                                                                                                                                                                    raw.set_channel_types(mapping)

                                                                                                                                                                                                                                                                                                                                                                        # raw.set_montage(mon)
                                                                                                                                                                                                                                                                                                                                                                            # picks
                                                                                                                                                                                                                                                                                                                                                                            # =
                                                                                                                                                                                                                                                                                                                                                                            # picks[10:30]
                                                                                                                                                                                                                                                                                                                                                                                # data
                                                                                                                                                                                                                                                                                                                                                                                # =
                                                                                                                                                                                                                                                                                                                                                                                # raw.get_data(picks,
                                                                                                                                                                                                                                                                                                                                                                                # start,
                                                                                                                                                                                                                                                                                                                                                                                # end)
                                                                                                                                                                                                                                                                                                                                                                                    events, times, corr = get_events(filename, au_emote_dict, classifier_loc,
                                                                                                                                                                                                                                                                                                                                                                                                                                                          real_time_file)
                                                                                                                                                                                                                                                                                                                                                                                        predicDic = {time: predic for time, predic in zip(times, corr)}
                                                                                                                                                                                                                                                                                                                                                                                            eventTimes = set(x[0] for x in events)
                                                                                                                                                                                                                                                                                                                                                                                                picks = mne.pick_types(raw.info, ecog=True)
                                                                                                                                                                                                                                                                                                                                                                                                    temp_one_data, temp_zero_data = get_window_data(raw, times, picks,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        eventTimes)
                                                                                                                                                                                                                                                                                                                                                                                                        one_data.extend(temp_one_data)
                                                                                                                                                                                                                                                                                                                                                                                                            zero_data.extend(temp_zero_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_emotion')
    parser.add_argument('-e', required=True, help="Path to edf directory")
    parser.add_argument(
        '-c', required=True, help='Name of computer for labeling')
    parser.add_argument('-au', required=True, help='Path to au_emote_dict')
    parser.add_argument('-cl', required=True, help='Path to classifier')
    parser.add_argument('-rf', required=True, help='Path to real time file')
    args = parser.parse_args()
    EDF_DIR = args['-e']
    MY_COMP = args['-c']
    AU_EMOTE_DICT_LOC = args['-au']
    CLASSIFIER_LOC = args['-cl']
    REAL_TIME_FILE_LOC = args['-rf']
    m = multiprocessing.Manager()
    zero_data = m.list()  # type: List[List[float]]
    one_data = m.list()  # type: List[List[float]]
    filenames = glob(os.path.join(EDF_DIR, '**/*.edf'), recursive=True)
    f = functools.partial(find_filename_data, AU_EMOTE_DICT_LOC, one_data,
                          zero_data, CLASSIFIER_LOC, REAL_TIME_FILE_LOC)
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
    np.save('classifier_data/all_{0}_data.npy'.format(MY_COMP), all_data)
    np.save('classifier_data/all_{0}_labels.npy'.format(MY_COMP), all_labels)
