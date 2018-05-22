import matplotlib
matplotlib.use('agg')
from threading import Thread
import argparse
from collections import deque
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
import matplotlib.pyplot as plt
from glob import glob
from chest import Chest
from mne.io import read_raw_edf
from welch import get_events
from pathos.multiprocessing import ProcessingPool as Pool
import functools
from tqdm import tqdm
from typing import List, Dict
from scipy.signal import welch
from contextlib import contextmanager
# from memory_profiler import profile
from dask.diagnostics import ProgressBar
from matplotlib.dates import DateFormatter


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def clean_times(times: deque, prev_pos: datetime.datetime,
                real_pos: datetime.datetime, emote_window_end):
    num_times = 0
    has_been_annotated = False

    for time in times:
        if prev_pos <= time < emote_window_end:
            num_times += 1
            has_been_annotated = True

        if time >= emote_window_end:
            break

    while times:
        time = times[0]

        if time <= real_pos:
            times.popleft()
        else:
            break

    return num_times, has_been_annotated


def get_window_data(raw: mne.io.Raw,
                    times,
                    corr,
                    picks,
                    eventTimes,
                    tqdm_num,
                    filename,
                    return_plot_data=False,
                    event_delta_seconds=1) -> tuple:
    ECOG_SAMPLING_FREQUENCY = 1000  # ECoG samples at a rate of 1000 Hz
    EVENT_DELTA_SECONDS = event_delta_seconds
    all_times = len(raw)
    times = deque(sorted(times))
    # times_corr = deque(sorted(zip(times, corr), key=lambda x: x[0]))

    if return_plot_data:
        plot_times = []
        plot_probs = []
    EVENT_THRESHOLD = .25

    # for time in times:
    # if time in eventTimes:
    # plot_ones.append[time]
    # else:
    # plot_zeros.append[time]

    eventTimes = deque(sorted(list(eventTimes)))
    ecog_start_time = sorted(times)[0]
    all_data = None
    labels = None
    range_times = np.arange(all_times)
    split_range_times = np.array_split(
        range_times,
        int(
            len(range_times) /
            (EVENT_DELTA_SECONDS * ECOG_SAMPLING_FREQUENCY)))
    freqs = None

    for index, split_time in enumerate(
            tqdm(split_range_times, position=tqdm_num, desc=filename)):

        if not times:
            break
        prev_range = split_range_times[index - 1]
        ecog_time_arr_prev = prev_range[0]
        ecog_time_arr_start = split_time[0]
        ecog_time_arr_end = split_time[len(split_time) - 1]
        # print(ecog_time_arr_start)
        curr_pos = ecog_start_time + \
            datetime.timedelta(seconds=ecog_time_arr_start /
                               ECOG_SAMPLING_FREQUENCY)
        prev_pos = ecog_start_time + \
            datetime.timedelta(seconds=ecog_time_arr_prev /
                               ECOG_SAMPLING_FREQUENCY)
        end_pos = ecog_start_time + \
            datetime.timedelta(seconds=ecog_time_arr_end /
                               ECOG_SAMPLING_FREQUENCY)
        # curr_corrs = []

        # curr_corrs.append(times_corr.popleft()[1])

        # for time in times:
        # if real_pos <= time <= emote_window_end:
        # has_been_annotated = True

        # break

        # if time > emote_window_end:
        # break
        num_times, has_been_annotated = clean_times(times, prev_pos, curr_pos,
                                                    end_pos)

        if not has_been_annotated:
            continue

        num_events, _ = clean_times(eventTimes, prev_pos, curr_pos, end_pos)

        prob = num_events / num_times
        has_event = prob >= EVENT_THRESHOLD

        if return_plot_data:
            plot_times.append(curr_pos)
            plot_probs.append(prob)
        label = 1 if has_event else 0
        # label = 1 if np.mean(curr_corrs) >= EVENT_THRESHOLD else 0
        data, ecog_times = raw[picks, ecog_time_arr_prev:ecog_time_arr_end]
        data = da.from_array(data, chunks=(1000, -1))
        psd = welch(data, 1000)

        if freqs is None:
            freqs = psd[0]

        psd_data = psd[1]
        try:
            psd_data = (psd_data - np.mean(psd_data, 1)[:, None]) / (np.std(
                psd_data, 1)[:, None])  # Rescale data
        except RuntimeWarning:
            continue

        # if np.allclose(psd_data, np.zeros(psd_data.shape)):
        # print('all 0')

        # continue

        if all_data is None:
            all_data = da.from_array(
                np.array([psd_data]), chunks=(10000, -1, -1))
        else:
            all_data = da.concatenate([all_data,
                                       np.array([psd_data])]).compute()

        if labels is None:
            labels = da.from_array(np.array([label]), chunks=(10000, ))
        else:
            labels = da.concatenate([labels, np.array([label])])
    # plot_dates = matplotlib.dates.date2num(plot_times)

    if return_plot_data:
        return plot_times, plot_probs
    else:
        return freqs, all_data, labels


def map_raw(filename: str):
    try:
        with suppress_stdout():
            raw = read_raw_edf(filename, preload=False)
    except ValueError:
        print('{0} has errors'.format(filename))

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

    return raw


def find_filename_data(au_emote_dict_loc,
                       classifier_loc,
                       real_time_file_loc,
                       out_loc,
                       out_q,
                       filename,
                       return_plot_data=False,
                       event_delta_seconds=1):
    tqdm_num, filename = filename
    tqdm_num = (tqdm_num % 5) + 1
    au_emote_dict = json.load(open(au_emote_dict_loc))
    raw = map_raw(filename)
    events, times, corr = get_events(filename, au_emote_dict, classifier_loc,
                                     real_time_file_loc)

    if times is None:
        times = []
    # print("{0} number of times: {1}".format(filename, len(times)))

    if times:
        predicDic = {time: predic for time, predic in zip(times, corr)}
        eventTimes = set(x[0] for x in events)
        picks = mne.pick_types(raw.info, ecog=True, ecg=True)

        if return_plot_data:
            return get_window_data(raw, times, corr, picks, eventTimes,
                                   tqdm_num, filename, return_plot_data,
                                   event_delta_seconds)

        freqs, temp_all_data, temp_labels = get_window_data(
            raw, times, corr, picks, eventTimes, tqdm_num, filename,
            return_plot_data, event_delta_seconds)

        if freqs is not None:
            freqs = da.from_array(freqs, chunks=(100, ))

        if freqs is not None:
            filename_out_dir = os.path.join(out_loc, 'classifier_data',
                                            os.path.basename(filename).replace(
                                                '.edf', ''))

            if not os.path.exists(filename_out_dir):
                os.makedirs(filename_out_dir)

            da.to_npy_stack(os.path.join(filename_out_dir, 'freqs'), freqs)

            if temp_all_data is not None:
                da.to_npy_stack(
                    os.path.join(filename_out_dir, 'data'),
                    da.from_array(temp_all_data, chunks=(10000, -1, -1)))

            if temp_labels is not None:
                da.to_npy_stack(
                    os.path.join(filename_out_dir, 'labels'),
                    da.from_array(temp_labels, chunks=(10000, )))
    out_q.put((filename, len(times)))


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
            with open(fn, 'r') as f:
                curr_arr = json.load(f)

                if m not in curr_arr:
                    curr_arr[m[0]] = m[1]

            with open(fn, 'w') as f:
                json.dump(curr_arr, f)
                f.flush()
    # f.close()


def clean_base(fn):
    return os.path.basename(fn).replace('.edf', '')


def get_args():
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

    return args


if __name__ == '__main__':
    args = get_args()
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
        already_done_dict = {}
        json.dump(already_done_dict, open(ALREADY_DONE_FILE, 'w'))
    m = multiprocessing.Manager()
    zero_data = m.list()  # type: List[List[float]]
    one_data = m.list()  # type: List[List[float]]
    filenames = glob(os.path.join(EDF_DIR, '**/*.edf'), recursive=True)
    already_done_dirs = os.listdir(
        os.path.join(OUT_FILE_PATH, 'classifier_data'))
    filenames = [
        x
        for x in clean_filenames(filenames, json.load(open(AU_EMOTE_DICT_LOC)))
        if (clean_base(x) not in already_done_dirs and (
            x not in already_done_dict or (already_done_dict[x] > 0)))
    ]
    out_q = m.Queue()

    Thread(target=listener, args=(ALREADY_DONE_FILE, out_q)).start()

    # for filename in tqdm(filenames):
    # find_filename_data(AU_EMOTE_DICT_LOC, CLASSIFIER_LOC,
    # REAL_TIME_FILE_LOC, OUT_FILE_PATH, out_q, filename)

    f = functools.partial(find_filename_data, AU_EMOTE_DICT_LOC,
                          CLASSIFIER_LOC, REAL_TIME_FILE_LOC, OUT_FILE_PATH,
                          out_q)
    num_processes = 5
    with tqdm(total=len(filenames)) as pbar:
        p = Pool(num_processes)

    for iteration, _ in enumerate(p.uimap(f, enumerate(filenames))):
        pbar.update()

    # for filename in tqdm(filenames):
    # find_filename_data(AU_EMOTE_DICT_LOC, CLASSIFIER_LOC,
    # REAL_TIME_FILE_LOC, OUT_FILE_PATH, out_q,
    # (0, filename))
    out_q.put('kill')
    p.close()
    p.join()
