import mne
import numpy as np
from mayavi import mlab
from mne import Evoked, evoked
from mne.viz import plot_alignment, snapshot_brain_montage
import matplotlib.pyplot as plt
from scipy.io import loadmat

evoked_arr = evoked.read_evokeds('test-ave.fif')
subjects_dir = mne.datasets.sample.data_path() + '/subjects'

fig = plot_alignment(evoked_arr[0].info, subject='sample', subjects_dir=subjects_dir, surfaces=['pial'])
mlab.view(200, 70)
info = evoked_arr[0].info
xy, im = snapshot_brain_montage(fig, info)

# Convert from a dictionary to array to plot
xy_pts = np.vstack(xy[ch] for ch in info['ch_names'])

# Define an arbitrary "activity" pattern for viz
activity = np.linspace(100, 200, xy_pts.shape[0])

# This allows us to use matplotlib to create arbitrary 2d scatterplots
fig2, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm')
ax.set_axis_off()
fig2.savefig('./brain.png', bbox_inches='tight')
