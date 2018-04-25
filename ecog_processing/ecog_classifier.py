import os
import dask.array as da
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
import h5py


def elbow_curve(data):
    components = range(1, data.shape[1] + 1)
    explained_variance = []

    for component in components:
        explained_variance.append(PCA(component).explained_variance_ratio)
    sns_plot = sns.regplot(
        x=components, y=explained_variance, fit_reg=False).get_figure()
    sns_plot.savefig("pca_elbow.png")


def get_data(data_loc: str):
    data_folders = os.listdir(data_loc)
    out_zeros = None
    out_ones = None

    # for data_file in data_files:
    # x = h5py.File(data_file)

    # if '/0' in x:
    # print(x['/0'])

    for data_folder in data_folders:
        zero_folder = os.path.join(data_folder, '0')
        one_folder = os.path.join(one_folder, '1')

        if os.path.exists(zero_folder):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_classifier')
    parser.add_argument('-d', required=True, help="Path to data")
    args = vars(parser.parse_args())
    DATA_LOC = args['d']
    get_data(DATA_LOC)
    # elbow_curve(get_data(DATA_LOC))
