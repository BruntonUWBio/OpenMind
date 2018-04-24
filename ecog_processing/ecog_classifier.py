import numpy as np
import sklearn.decomposition.PCA as PCA
import seaborn as sns
import argparse


def elbow_curve(data: np.ndarray):
    components = range(1, data.shape[1] + 1)
    explained_variance = []

    for component in components:
        explained_variance.append(PCA(component).explained_variance_ratio)
    sns_plot = sns.regplot(
        x=components, y=explained_variance, fit_reg=False).get_figure()
    sns_plot.savefig("pca_elbow.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_classifier')
    parser.add_argument('-d', required=True, help="Path to data")
    args = vars(parser.parse_args())
    DATA_LOC = args['d']
    elbow_curve(np.load(DATA_LOC))
