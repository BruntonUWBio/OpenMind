from ecog_emotion import find_filename_data, get_args, clean_filenames
import os
from glob import glob
import json

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

    for filename in filenames:
        if 'cb46fd46' in filename:
            times, probs = find_filename_data(
                AU_EMOTE_DICT_LOC, CLASSIFIER_LOC, REAL_TIME_FILE_LOC, OUT_FILE_PATH, out_q, filename, True, 60)
            print(times)
            print(probs)
