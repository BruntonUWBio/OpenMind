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
import sklearn

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
def run_tpot(zeros, ones):
    y = [0 for x in zeros]
    y.extend([1 for x in ones])
    all_data = da.concatenate([zeros, ones]).compute()
    all_data = all_data.reshape(all_data.shape[0], all_data.shape[1] * all_data.shape[2])
    pca = PCA(n_components=5)
    all_data = pca.fit_transform(np.nan_to_num(all_data))
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, y, test_size=.1, stratify=y)

    if not os.path.exists('tpot_checkpoint'):
        os.mkdir('tpot_checkpoint')

    # tpot = TPOTClassifier(
        # n_jobs=-1,
        # verbosity=3,
        # scoring='f1',
        # # subsample=.5,
        # periodic_checkpoint_folder='tpot_checkpoint',
        # max_eval_time_mins=20,
        # memory='auto')
    exported_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        PCA(iterated_power=8, svd_solver="randomized"),
        StackingEstimator(estimator=LinearSVC(C=0.01, dual=True, loss="squared_hinge", penalty="l2", tol=0.1)),
        ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=17, min_samples_split=6, n_estimators=100)
    )

    exported_pipeline.fit(X_train, y_train)
    results = exported_pipeline.predict(X_test)
    # tpot.fit(X_train, y_train)
    # tpot.export('tpot_ecog_pipeline.py')
    out_file = open('tpot_test.txt', 'w')
    out_file.write(sklearn.metrics.classification_report(y_test, results))

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
    data_folders = [os.path.join(data_loc, x) for x in os.listdir(data_loc) if 'cb46fd46' in x]
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
