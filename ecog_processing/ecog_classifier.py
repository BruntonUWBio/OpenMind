import os
import dask.array as da
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
import h5py
from tqdm import tqdm
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split


def run_tpot(zeros, ones):
    y = [0 for x in zeros]
    y.extend([1 for x in ones])
    all_data = da.concatenate([zeros, ones]).compute()
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, y, test_size=.33)
    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    if not os.path.exists('tpot_checkpoint'):
        os.mkdir('tpot_checkpoint')

    if not os.path.exists('tpot_cache'):
        os.mkdir('tpot_cache')
    tpot = TPOTClassifier(
        n_jobs=-1,
        verbosity=3,
        scoring='f1',
        subsample=.5,
        periodic_checkpoint_folder='tpot_checkpoint',
        max_eval_time_mins=20,
        memory='tpot_cache')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_ecog_pipeline.py')


def elbow_curve(data):
    data = data[1]
    components = range(1, data.shape[1] + 1)
    explained_variance = []

    for component in tqdm(components[:50]):
        pca = PCA(n_components=component)
        pca.fit(data)
        explained_variance.append(sum(pca.explained_variance_ratio_))
    sns_plot = sns.regplot(
        x=components[:50], y=explained_variance, fit_reg=False).get_figure()
    sns_plot.savefig("pca_elbow.png")


def get_data(data_loc: str) -> tuple:
    data_folders = [os.path.join(data_loc, x) for x in os.listdir(data_loc)]
    out_zeros = None
    out_ones = None

    for data_folder in tqdm(data_folders):
        zero_folder = os.path.join(data_folder, '0')
        one_folder = os.path.join(data_folder, '1')

        if os.path.exists(zero_folder):
            if out_zeros is None:
                out_zeros = da.from_npy_stack(zero_folder)
            else:
                out_zeros = da.concatenate(
                    [out_zeros, da.from_npy_stack(zero_folder)])

        if os.path.exists(one_folder):
            if out_ones is None:
                out_ones = da.from_npy_stack(one_folder)
            else:
                out_ones = da.concatenate(
                    [out_ones, da.from_npy_stack(one_folder)])

    return out_zeros.compute(), out_ones.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_classifier')
    parser.add_argument('-d', required=True, help="Path to data")
    args = vars(parser.parse_args())
    DATA_LOC = args['d']
    zeros, ones = get_data(DATA_LOC)
    run_tpot(zeros, ones)
    # elbow_curve(get_data(DATA_LOC))
