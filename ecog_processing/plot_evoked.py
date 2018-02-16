import os
import sys

from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from mne.time_frequency import psd_welch

sys.path.append('/home/gauthv/PycharmProjects/ecogAnalysis')
from ecog_processing.viewSTLmayavi import get_mayavi_fig

os.environ['ETS_TOOLKIT'] = 'wx'
# os.environ['QT_API'] = 'pyqt'

import mne
import numpy as np
from mayavi import mlab
from mne import Evoked, evoked, read_epochs
from mne.viz import plot_alignment, snapshot_brain_montage, epochs
import matplotlib.pyplot as plt
from scipy.io import loadmat


class UpdateBrain:
    def __init__(self, psds, freqs, xy_pts, ax):
        self.freqs = freqs
        self.psds = psds
        self.ax = ax
        self.xy_pts = xy_pts
        self.epoch_arr = epoch_arr
        ax.imshow(im)
        ax.set_axis_off()
        self.activity = self.get_first_activity()
        self.patches = ax.scatter(*xy_pts.T, c=self.activity, s=100, cmap='coolwarm',
                                  norm=Normalize().autoscale(self.activity))

    def init(self):
        self.patches.set_color('w')
        return self.patches,

    def get_first_activity(self):
        return np.zeros((self.xy_pts.shape[0],))

    def __call__(self, i):
        # if i == 0:
        #     return self.init()
        # else:
        curr_psds = self.psds[i]
        electrode_averages = np.mean(curr_psds, 1)

        map = cm.ScalarMappable(Normalize().autoscale(electrode_averages), cmap='coolwarm')
        rgbs = map.to_rgba(electrode_averages)
        self.patches.set_color(rgbs)
        # self.patches = self.ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm', norm=Normalize().autoscale(activity))
        return self.patches,


def create_animation(epoch_arr, xy_pts):
    psds, freqs = psd_welch(epoch_arr, 32, 100, n_jobs=2, verbose=True)
    fig2, ax = plt.subplots()
    ud = UpdateBrain(psds, freqs, xy_pts, ax)
    anim = FuncAnimation(fig2, ud, frames=np.arange(len(epoch_arr)), init_func=ud.init, interval=100, blit=True)
    anim.save('brain_anim.mp4')


if __name__ == '__main__':
    # evoked_arr = evoked.read_evokeds('test-ave.fif')
    epoch_arr = read_epochs('test-epo.fif')
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    # subjects_dir = '/home/gauthv/PycharmProjects/ecogAnalysis/'

    mlab_fig = get_mayavi_fig('../ecb43e/both_lowres.stl', 'trodes.mat')

    # path_data = mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat'
    path_data = 'trodes.mat'
    mat = loadmat(path_data)
    ch_names = epoch_arr[0].ch_names
    # elec = mat['elec'][:len(ch_names)]
    elec = mat['AllTrodes'][:len(ch_names)]
    dig_ch_pos = dict(zip(ch_names, elec))
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., 'ecog', montage=mon)
    # fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
    #                      surfaces=['pial'], meg=False)

    # info = epoch_arr[0].info
    # fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir, surfaces=['pial'])
    # fig = plot_alignment(info, subject='ecb43e', subjects_dir=subjects_dir, surfaces=['pial'])
    mlab.view(200, 80)
    xy, im = snapshot_brain_montage(mlab_fig, info)

    # Convert from a dictionary to array to plot
    xy_pts = np.stack(xy[ch] for ch in info['ch_names'])

    # Define an arbitrary "activity" pattern for viz
    activity = np.zeros((xy_pts.shape[0],))
    # data = epoch_arr.get_data()
    # for event in epoch_arr.events:

    create_animation(epoch_arr, xy_pts)

    # for evoked_epoch in epoch_arr.iter_evoked():
    #     # time = event[0]
    #     # epoch_cropped = epoch_arr.crop(time - 10000, time + 10000)
    #     for index, electrode in enumerate(evoked_epoch.data):
    #         activity[index] += np.mean([abs(x) for x in electrode])
    #
    # # This allows us to use matplotlib to create arbitrary 2d scatterplots
    # fig2, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(im)
    # ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm', norm=Normalize().autoscale(activity))
    # ax.set_axis_off()
    # fig2.savefig('./brain.png', bbox_inches='tight')
