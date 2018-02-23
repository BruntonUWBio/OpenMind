import numpy as np
from ecog_processing.plot_evoked import do_psd
from mne import read_epochs

if __name__ == '__main__':
    epoch_arr = read_epochs('test-epo.fif')
    psds, freqs = do_psd(epoch_arr)
    corr_arr = np.load('corr_arr.npy')
    print('ass')
