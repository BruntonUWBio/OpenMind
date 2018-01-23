import matplotlib.pyplot as plt
import mne
import datetime
import pandas as pd
import numpy as np
from mne.time_frequency import psd_welch

fig, ax = plt.subplots(1)
raw = mne.io.read_raw_edf('/data1/edf/a1d36553/a1d36553_8.edf', preload=False)

info = raw.info
basetime_posix = info['meas_date']

chanel1 = raw.get_data(picks=1, start=0, stop = 400000, reject_by_annotation=None, return_times= True)

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
picks = picks[10:25]

fmin, fmax = .1, 160
n_fft = 4096

tmin = raw.times[0]
tmax = raw.times[raw.n_times - 1]

psd, freqs = psd_welch(raw, fmin, fmax, n_fft=n_fft, tmin=2000, tmax=10000, n_jobs=-1, picks=picks)
cmap = 'RdBu_r'
freq_mask = freqs < 10
freqs = freqs[freq_mask]
log_psd = 10 * np.log10(psd)

im = ax.imshow(log_psd[:, freq_mask].T, aspect='auto', origin='lower', cmap=cmap)

ax.set_yticks(np.arange(0, len(freqs), 10))
ax.set_yticklabels(freqs[::10].round(1))
ax.set_ylabel('Frequency (Hz)')
ax.set_xticks(np.arange(0, len(picks)))
ax.set_xticklabels(picks)
ax.set_xlabel('EEG channel index')
im.set_clim()
plt.title('continous power spectrum from {0} to {1}'.format(tmin, tmax))
plt.show()
