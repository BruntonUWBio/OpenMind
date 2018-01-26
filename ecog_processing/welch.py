import mne
from mayavi import mlab
from mne.viz import plot_alignment
from scipy.io import loadmat

# fig, ax = plt.subplots(1)
# raw = mne.io.read_raw_edf('/data1/edf/a1d36553/a1d36553_8.edf', preload=False)

# info = raw.info
# basetime_posix = info['meas_date']

# chanel1 = raw.get_data(picks=1, start=0, stop = 400000, reject_by_annotation=None, return_times= True)
#
# picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
# picks = picks[10:25]
#
# fmin, fmax = .1, 160
# n_fft = 4096
#
# tmin = raw.times[0]
# tmax = raw.times[raw.n_times - 1]
#
# psd, freqs = psd_welch(raw, fmin, fmax, n_fft=n_fft, tmin=2000, tmax=10000, n_jobs=-1, picks=picks)
# cmap = 'RdBu_r'
# freq_mask = freqs < 10
# freqs = freqs[freq_mask]
# log_psd = 10 * np.log10(psd)
#
# im = ax.imshow(log_psd[:, freq_mask].T, aspect='auto', origin='lower', cmap=cmap)
#
# ax.set_yticks(np.arange(0, len(freqs), 10))
# ax.set_yticklabels(freqs[::10].round(1))
# ax.set_ylabel('Frequency (Hz)')
# ax.set_xticks(np.arange(0, len(picks)))
# ax.set_xticklabels(picks)
# ax.set_xlabel('EEG channel index')
# im.set_clim()
# plt.title('continous power spectrum from {0} to {1}'.format(tmin, tmax))
# plt.show()
mat = loadmat('/homes/iws/gauthv/trodes.mat')
ch_names = list(map(str, range(mat['Grid'].shape[0])))
elec = mat['Grid']
# mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
# ch_names = mat['ch_names'].tolist()
# elec = mat['elec']

dig_ch_pos = dict(zip(ch_names, elec))
mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos, point_names=ch_names)
# print('Created %s channel positions' % len(ch_names))
# info = mne.create_info(ch_names, 1000., 'ecog', montage=mon, verbose=3)
# subjects_dir = mne.datasets.sample.data_path() + '/subjects'
#
# fig = plot_alignment(info, subject='sample', meg=False, subjects_dir=subjects_dir, surfaces=['pial'], verbose=3)
# mlab.savefig('sample_alignment.png', figure=fig)
# # mlab.view(200, 70)

# mon.transform_to_head()
# mon.save('montage.fif')

evoked = mne.Evoked(verbose=3)
evoked.set_montage(mon)

