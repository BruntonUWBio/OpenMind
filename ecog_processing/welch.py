import mne
import pandas as pd
from mne.io import read_raw_edf

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


# mat = loadmat('/data/electrodes/cb46fd46/ecb43e/bis_trodes.mat')


# mat_array = mat['Montage']
# open('montage.txt', 'w').write(str(mat_array))
# file_mat_array = np.fromfile('montage.txt')


# ch_names = list(map(str, range(mat['Grid'].shape[0])))
# elec = mat['Grid']


# mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
# ch_names = mat['ch_names'].tolist()
# elec = mat['elec']

# dig_ch_pos = dict(zip(ch_names, elec))
# mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos, point_names=ch_names)
# print('Created %s channel positions' % len(ch_names))
# info = mne.create_info(ch_names, 1000., 'ecog', montage=mon, verbose=3)
# subjects_dir = mne.datasets.sample.data_path() + '/subjects'
#
# fig = plot_alignment(info, subject='sample', meg=False, subjects_dir=subjects_dir, surfaces=['pial'], verbose=3)
# mlab.savefig('sample_alignment.png', figure=fig)
# # mlab.view(200, 70)

# mon.transform_to_head()
# mon.save('montage.fif')

raw = read_raw_edf('/home/gvelchuru/a1d36553_8.edf')


# print(raw[0:100])
# events = mne.find_events(raw)
# raw.save('test.raw.fif')


# evoked = mne.Evoked(None, stim_cha)
# evoked.set_montage(mon)


def get_datetimes(raw, start, end):
    info = raw.info
    basetime_posix = info['meas_date']
    channel1 = raw.get_data(picks=1, start=start, stop=end, reject_by_annotation=None, return_times=True)
    datetimes = pd.to_datetime(channel1[1].ravel() + basetime_posix, unit='s')
    return datetimes


def get_events(filename, datetimes):
    events = None
    pass


if __name__ == '__main__':
    filename = '/home/gvelchuru/a1d36553_8.edf'
    raw = read_raw_edf(filename)
    start = 200000
    end = 400000
    datetimes = get_datetimes(raw, start, end)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    picks = picks[10:30]
    data = raw.get_data(picks, start, end)
    events = get_events(filename, datetimes)
