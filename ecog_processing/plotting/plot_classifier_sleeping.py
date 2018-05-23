import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecog_classifier import get_data, get_data_loc, make_all_data
from dask import array as da
import pudb


def run_pipeline(data, labels, times):
    data_plus_times = da.concatenate(
        [data, times.reshape((times.size, 1))], axis=1).compute()
    tpot = make_pipeline(
        PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=False),
        PCA(iterated_power=8, svd_solver="randomized"),
        StackingEstimator(
            estimator=LinearSVC(
                C=0.01, dual=True, loss="squared_hinge", penalty="l2",
                tol=0.1)),
        ExtraTreesClassifier(
            bootstrap=False,
            criterion="gini",
            max_features=0.9000000000000001,
            min_samples_leaf=17,
            min_samples_split=6,
            n_estimators=100))
    X_train, X_test, y_train, y_test = train_test_split(
        data_plus_times, labels, test_size=.25)
    X_train = X_train[:, :X_train.shape[1] - 1]
    test_times = X_test[:, X_test.shape[1] - 1]
    X_test = X_test[:, :X_test.shape[1] - 1]
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    tpot.fit(X_train, y_train)

    time_dict = {}  # type: Dict[datetime, List[int]]

    # pudb.set_trace()

    for test_data, test_time in zip(X_test, test_times):
        test_time = datetime.fromtimestamp(test_time)
        predicted = tpot.predict(test_data.reshape(1, -1))[0]
        found_pair = False

        for existing_date in time_dict:
            if existing_date.hour == test_time.hour and existing_date.day == test_time.day and existing_date.minute == test_time.minute:
                time_dict[existing_date][predicted] += 1
                found_pair = True

                break

        if not found_pair:
            time_dict[test_time] = [0, 0]
            time_dict[test_time][predicted] += 1

    plot_times = []
    not_happies = []
    happies = []

    for time, data in time_dict.items():
        data_sum = sum(data)
        # time_dict[time] = [data[0] / data_sum, data[1] / data_sum]
        plot_times.append(time)
        not_happies.append(data[0] / data_sum)
        happies.append(data[1] / data_sum)

    trace1 = go.Bar(x=plot_times, y=not_happies, name='Not Happy')
    trace2 = go.Bar(x=plot_times, y=happies, name='Happy')

    data = [trace1, trace2]
    layout = go.Layout(barmode='stack')

    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='classification_by_time_of_day', auto_open=False)


if __name__ == '__main__':
    DATA_LOC = get_data_loc()
    data, labels, times = get_data(DATA_LOC)
    data, labels = make_all_data(data, labels)
    run_pipeline(data, labels, times)
