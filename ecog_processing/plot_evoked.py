import os
import sys

from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from mne.time_frequency import psd_welch

from viewSTLmayavi import get_mayavi_fig

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
        curr_psds = self.psds[i]
        electrode_averages = np.mean(curr_psds, 1)

        map = cm.ScalarMappable(Normalize().autoscale(electrode_averages), cmap='coolwarm')
        rgbs = map.to_rgba(electrode_averages)
        self.patches.set_color(rgbs)
        return self.patches,


def do_psd(epoch_arr):
    return psd_welch(epoch_arr, 32, 100, n_jobs=2, verbose=True)


def create_animation(epoch_arr, xy_pts):
    psds, freqs = do_psd(epoch_arr)
    fig2, ax = plt.subplots()
    ud = UpdateBrain(psds, freqs, xy_pts, ax)
    anim = FuncAnimation(fig2, ud, frames=np.arange(len(epoch_arr)), init_func=ud.init, interval=100, blit=True)
    anim.save('brain_anim.mp4')


if __name__ == '__main__':
    trodes_mat = sys.argv[sys.argv.index('-t') + 1]
    # evoked_arr = evoked.read_evokeds('test-ave.fif')
    epoch_arr = read_epochs('test-epo.fif')
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    # subjects_dir = '/home/gauthv/PycharmProjects/ecogAnalysis/'

    mlab_fig = get_mayavi_fig('../ecb43e/both_lowres.stl', trodes_mat)

    # path_data = mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat'

    mat = loadmat(trodes_mat)
    ch_names = [x for x in epoch_arr[0].ch_names if 'GRID' in x]
    # elec = mat['elec'][:len(ch_names)]
    elec = mat['Grid']
    dig_ch_pos = dict(zip(ch_names, elec))

    mlab.view(210, 90)
    xy, im = snapshot_brain_montage(mlab_fig, epoch_arr.info)

    # Convert from a dictionary to array to plot
    xy_pts = np.stack(xy[ch] for ch in epoch_arr.info['ch_names'])

    activity = np.zeros((xy_pts.shape[0],))

    create_animation(epoch_arr, xy_pts)
