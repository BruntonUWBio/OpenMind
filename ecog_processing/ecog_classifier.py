import sklearn
import dask
import dask.array as da


def pca_elbow_curve(data: dask.array):
