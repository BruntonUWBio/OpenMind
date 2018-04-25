from threading import Thread
import argparse
import json
import random
import sys
import gc
import os
import datetime
import multiprocessing
import dask
import dask.array as da
import sys
import os
import numpy as np
import mne
from glob import glob
from mne.io import read_raw_edf
from welch import get_events
from pathos.multiprocessing import ProcessingPool as Pool
import functools
from tqdm import tqdm
from typing import List, Dict
from scipy.signal import welch
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_window_data(raw: mne.io.Raw, times: list, picks,
                    eventTimes: set) -> tuple:
    ECOG_SAMPLING_FREQUENCY = 1000  # ECoG samples at a rate of 1000 Hz
    EVENT_DELTA_SECONDS = 1
    all_times = len(raw)
    ecog_start_time = sorted(times)[0]
    one_data = None
    zero_data = None
    range_times = np.arange(all_times)
    split_range_times = np.array_split(
        range_times, int(len(range_times) / (ECOG_SAMPLING_FREQUENCY)))
    freqs = None

    for split_time in split_range_times:
        gc.collect()
        ecog_time_arr_start = split_time[0]
        ecog_time_arr_end = split_time[len(split_time) - 1]
        real_pos = ecog_start_time + \
            datetime.timedelta(seconds=ecog_time_arr_start /
                               ECOG_SAMPLING_FREQUENCY)
        emote_window_end = real_pos + \
            datetime.timedelta(seconds=ecog_time_arr_end /
                               ECOG_SAMPLING_FREQUENCY)
        has_event = False
        has_been_annotated = False

        for time in times:
            if real_pos <= time <= emote_window_end:
                has_been_annotated = True

                break

            if time > emote_window_end:
                break

        if not has_been_annotated:
            continue

        for event_time in eventTimes:
            if real_pos <= event_time <= emote_window_end:
                has_event = True

                break

            if event_time > emote_window_end:
                break
        # try:
        data, ecog_times = raw[picks, ecog_time_arr_start:ecog_time_arr_end]
        data = da.from_array(data, chunks=(100, -1))
        psd = welch(data)
        # print("DIFFERENCE BETWEEN ECOG TIMES IS: {0}".format(
        # ecog_time_arr_end - ecog_time_arr_start))

        if freqs is None:
            freqs = psd[0]

        if has_event:
            if one_data is None:
                one_data = da.from_array(psd[1], chunks=(1000, -1))
            else:
                one_data = da.concatenate([one_data, psd[1]])
        else:
            if zero_data is None:
                zero_data = da.from_array(psd[1], chunks=(1000, -1))
            else:
                zero_data = da.concatenate([zero_data, psd[1]])

    print("LEN ONES: {0}".format(one_data.shape))
    print("LEN ZEROS: {0}".format(zero_data.shape))

    return freqs, one_data, zero_data


def find_filename_data(au_emote_dict_loc, classifier_loc, real_time_file_loc,
                       out_loc, out_q, filename):
    au_emote_dict = json.load(open(au_emote_dict_loc))
    try:
        with suppress_stdout():
            raw = read_raw_edf(filename, preload=False)
    except ValueError:
        return
    mapping = {
        ch_name: 'ecog'
        for ch_name in raw.ch_names if 'GRID' in ch_name
    }  # type: Dict[str, str]
    mapping.update(
        {ch_name: 'ecg'
         for ch_name in raw.ch_names if 'ECG' in ch_name})
    mapping.update(
        {ch_name: 'eeg'
         for ch_name in raw.ch_names if ch_name not in mapping})

    if 'ecog' not in mapping.values():
        return

    raw.set_channel_types(mapping)

    events, times, corr = get_events(filename, au_emote_dict, classifier_loc,
                                     real_time_file_loc)

    if times is None:
        times = []
    print("{0} number of times: {1}".format(filename, len(times)))

    if times:
        predicDic = {time: predic for time, predic in zip(times, corr)}
        eventTimes = set(x[0] for x in events)
        picks = mne.pick_types(raw.info, ecog=True)
        freqs, temp_one_data, temp_zero_data = get_window_data(
            raw, times, picks, eventTimes)

        if freqs is not None:
            freqs = da.from_array(freqs, chunks=(100, ))

        if freqs is not None:
            filename_out_dir = os.path.join(out_loc, 'classifier_data',
                                            os.path.basename(filename).replace(
                                                '.edf', ''))

            if not os.path.exists(filename_out_dir):
                os.makedirs(filename_out_dir)

            da.to_npy_stack(os.path.join(filename_out_dir, 'freqs'), freqs)

            if temp_zero_data is not None:
                da.to_npy_stack(
                    os.path.join(filename_out_dir, '0'), temp_zero_data)

            if temp_one_data is not None:
                da.to_npy_stack(
                    os.path.join(filename_out_dir, '1'), temp_one_data)
    out_q.put(filename)


def clean_filenames(filenames: list, au_dict: dict) -> list:
    out_names = []

    for filename in filenames:
        patient_session = os.path.basename(filename).replace('.edf', '')

        for vid_num in au_dict:
            if patient_session in vid_num:
                out_names.append(filename)

                break

    return out_names


def listener(fn, q):
    '''listens for messages on the q, writes to file. '''

    while 1:
        if not q.empty():
            m = q.get()

            if m == 'kill':
                break
            f = open(fn, 'r')
            curr_arr = json.load(f)
            curr_arr.append(m)
            f = open(fn, 'w')
            json.dump(curr_arr, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_emotion')
    parser.add_argument('-e', required=True, help="Path to edf directory")
    parser.add_argument(
        '-c', required=True, help='Name of computer for labeling')
    parser.add_argument('-au', required=True, help='Path to au_emote_dict')
    parser.add_argument('-cl', required=True, help='Path to classifier')
    parser.add_argument('-rf', required=True, help='Path to real time file')
    parser.add_argument('-o', required=True, help='Out file path')
    parser.add_argument(
        '-a', required=True, help='Parent directory of already_done_file')
    args = vars(parser.parse_args())
    EDF_DIR = args['e']
    MY_COMP = args['c']
    AU_EMOTE_DICT_LOC = args['au']
    CLASSIFIER_LOC = args['cl']
    REAL_TIME_FILE_LOC = args['rf']
    OUT_FILE_PATH = args['o']
    ALREADY_DONE_FILE = os.path.join(args['a'], 'already_done.txt')

    if os.path.exists(ALREADY_DONE_FILE):
        already_done_dict = json.load(open(ALREADY_DONE_FILE))
    else:
        already_done_dict = []
        json.dump(already_done_dict, open(ALREADY_DONE_FILE, 'w'))
    m = multiprocessing.Manager()
    zero_data = m.list()  # type: List[List[float]]
    one_data = m.list()  # type: List[List[float]]
    filenames = glob(os.path.join(EDF_DIR, '**/*.edf'), recursive=True)
    filenames = [
        x
        for x in clean_filenames(filenames, json.load(open(AU_EMOTE_DICT_LOC)))
        if x not in already_done_dict
    ]
    out_q = m.Queue()
    f = functools.partial(find_filename_data, AU_EMOTE_DICT_LOC,
                          CLASSIFIER_LOC, REAL_TIME_FILE_LOC, OUT_FILE_PATH,
                          out_q)
    with tqdm(total=len(filenames)) as pbar:
        p = Pool(4)

    Thread(target=listener, args=(ALREADY_DONE_FILE, out_q)).start()
    # watcher = p.apply_async(listener, (ALREADY_DONE_FILE, out_q))

    for iteration, _ in enumerate(p.uimap(f, filenames)):
        pbar.update()

    out_q.put('kill')
    p.close()
