from ecog_emotion import find_filename_data, get_args, clean_filenames
import os
from glob import glob
from queue import Queue
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = get_args()
    EDF_DIR = args['e']
    MY_COMP = args['c']
    AU_EMOTE_DICT_LOC = args['au']
    CLASSIFIER_LOC = args['cl']
    REAL_TIME_FILE_LOC = args['rf']
    OUT_FILE_PATH = args['o']
    ALREADY_DONE_FILE = os.path.join(args['a'], 'already_done.txt')
    filenames = glob(os.path.join(EDF_DIR, '**/*.edf'), recursive=True)
    filenames = [
        x
        for x in clean_filenames(filenames, json.load(open(AU_EMOTE_DICT_LOC)))
    ]

    out_q = Queue()

    if not os.path.exists(OUT_FILE_PATH):
        os.mkdir(OUT_FILE_PATH)

    for filename in filenames:
        if 'cb46fd46' in filename:
            returnVal = find_filename_data(AU_EMOTE_DICT_LOC, CLASSIFIER_LOC,
                                           REAL_TIME_FILE_LOC, OUT_FILE_PATH,
                                           out_q, (0, filename), True, 60)

            if returnVal is not None:
                times, probs = returnVal
                f, ax1 = plt.subplots()
                times = times[:3]
                probs = probs[:3]
                ax1.bar(times, probs, label='Happy', width=1)
                ax1.bar(
                    times, [1 - x for x in probs],
                    bottom=probs,
                    label='Not Happy',
                    width=1)
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Probability")
                plt.legend()
                plt.savefig(
                    os.path.join(OUT_FILE_PATH,
                                 os.path.basename(filename).replace(
                                     '.edf', '') + '.png'))
                plt.clf()
                plt.cla()
