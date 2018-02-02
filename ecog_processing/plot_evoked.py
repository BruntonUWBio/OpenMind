import os

os.environ['ETS_TOOLKIT'] = 'wx'
# os.environ['QT_API'] = 'pyqt'

import mne
import numpy as np
from mayavi import mlab
from mne import Evoked, evoked, read_epochs
from mne.viz import plot_alignment, snapshot_brain_montage, epochs
import matplotlib.pyplot as plt
from scipy.io import loadmat

# evoked_arr = evoked.read_evokeds('test-ave.fif')
epoch_arr = read_epochs('test-epo.fif')
subjects_dir = mne.datasets.sample.data_path() + '/subjects'

# path_data = mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat'
# mat = loadmat(path_data)
# ch_names = epoch_arr[0].ch_names
# elec = mat['elec'][:len(ch_names)]
# dig_ch_pos = dict(zip(ch_names, elec))
# mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
# info = mne.create_info(ch_names, 1000., 'ecog', montage=mon)
# fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
#                      surfaces=['pial'], meg=False)

info = epoch_arr[0].info
fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir, surfaces=['pial'])
mlab.view(200, 70)
xy, im = snapshot_brain_montage(fig, info)

# Convert from a dictionary to array to plot
xy_pts = np.stack(xy[ch] for ch in info['ch_names'])

# Define an arbitrary "activity" pattern for viz
activity = np.linspace(100, 200, xy_pts.shape[0])

# This allows us to use matplotlib to create arbitrary 2d scatterplots
fig2, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm')
ax.set_axis_off()
fig2.savefig('./brain.png', bbox_inches='tight')
