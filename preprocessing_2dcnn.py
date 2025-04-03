import numpy as np
import pandas as pd
from scipy.signal import spectrogram

#STFT spectrogram
def convert_epoch_to_spectrogram(epoch_row, channels, fs=250, nperseg=128, noverlap=64):
    """
    Given a pandas Series representing one epoch, where each channel column contains
    a 1D numpy array of time series data, compute a spectrogram for each channel and
    stack them into a 3D array of shape (n_channels, freq_bins, time_bins).
    
    Parameters:
      epoch_row: pandas Series
          One row of your preprocessed dataframe (one epoch).
      channels: list of str
          The list of channel names to process.
      fs: int
          Sampling frequency (default 250 Hz).
      nperseg: int
          Length of each segment for spectrogram calculation.
      noverlap: int
          Number of overlapping samples between segments.
    
    Returns:
      spec_stack: numpy array
          A 3D array with shape (n_channels, freq_bins, time_bins) containing the
          spectrogram (in dB) for each channel.
    """
    spec_list = []
    for ch in channels:
        ts = epoch_row[ch]  # this is the 1D time series for the channel
        # Compute the spectrogram
        f, t, Sxx = spectrogram(ts, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Convert the power spectrogram to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        spec_list.append(Sxx_db)
    spec_stack = np.stack(spec_list, axis=0)
    return spec_stack

#CWT scalogram


def convert_preprocessed_df_to_2d(preprocessed_df, channels=["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF"], fs=250, nperseg=128, noverlap=64):

    # Make a copy to avoid modifying the original dataframe
    df = preprocessed_df.copy()
    
    # Compute the spectrogram for each row (epoch)
    df["spectrogram"] = df.apply(lambda row: convert_epoch_to_spectrogram(row, channels, fs, nperseg, noverlap), axis=1)
    
    #drop channel columns and other useless ones
    df_final = df[["spectrogram", "epilepsy", "age", "gender",'subject_id', 'edf_path']]
    return df_final


