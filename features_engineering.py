# EEG Feature Extraction from Preprocessed TUH EEG Segments
# ----------------------------------------------------------
# This script extracts handcrafted features from 5-second EEG segments that were preprocessed earlier.
# Each row in the input represents one segment (flattened) with arrays for 5 EEG channels.
# 
# - Each signal is analyzed per channel (P3, T3, C3, Cz, C4)
# - We extract:
#     - Time-domain statistics: mean, var, skew, kurtosis, zero-crossing rate (ZCR), Teager energy
#     - Frequency-domain: relative bandpower (delta, theta, alpha, beta, gamma)
#     - Complexity/entropy: sample entropy, permutation entropy
#     - Wavelet domain: energy, variance, entropy of each level from DWT (db4, level 4)
# - Features are prefixed per channel and merged with metadata (age, gender, label, subject_id)
# 
# Final output is a DataFrame with one row per segment (same shape as input), and features grouped by channel.
# 
# Output file: 'eeg_features_updated.csv'

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy import sample_entropy, perm_entropy
import pywt

# Frequency bands of interest for epilepsy detection
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

WAVELET = 'db4'
DWT_LEVEL = 4

# Compute relative band power using Welch's method
def bandpower(data, sf, band, relative=True):
    low, high = band
    nperseg = min(256, len(data))
    freqs, psd = welch(data, sf, nperseg=nperseg)
    idx = np.logical_and(freqs >= low, freqs <= high)
    power = np.trapz(psd[idx], freqs[idx])
    if relative:
        total = np.trapz(psd, freqs)
        power = power / total if total > 0 else 0
    return power

# Main function to compute features from a single channel signal
def compute_features(signal, sf=250):
    features = {}
    features['mean'] = np.mean(signal)
    features['var'] = np.var(signal)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal, fisher=False)
    features['zcr'] = np.mean(np.diff(np.sign(signal)) != 0)
    features['tkeo'] = np.mean(signal[1:-1]**2 - signal[:-2]*signal[2:]) if len(signal) > 2 else 0

    for name, (lo, hi) in BANDS.items():
        features[f'{name}_power'] = bandpower(signal, sf, (lo, hi))

    features['samp_entropy'] = sample_entropy(signal, 2, 0.2 * np.std(signal))
    features['perm_entropy'] = perm_entropy(signal, order=3, normalize=True)

    coeffs = pywt.wavedec(signal, wavelet=WAVELET, level=DWT_LEVEL)
    for i, c in enumerate(coeffs):
        band = f"A{DWT_LEVEL}" if i == 0 else f"D{DWT_LEVEL - i + 1}"
        energy = np.sum(c**2)
        var = np.var(c)
        p = np.abs(c)**2
        p /= (np.sum(p) + 1e-12)
        shannon = -np.sum(p * np.log2(p + 1e-12))
        features[f'{band}_energy'] = energy
        features[f'{band}_var'] = var
        features[f'{band}_entropy'] = shannon

    return features

# Main extraction function

def extract_all_features(df, output_csv='eeg_features_updated.csv'):
    # Channel order must match preprocessing
    channel_names = ["EEG P3-REF", "EEG T3-REF", "EEG C3-REF", "EEG CZ-REF", "EEG C4-REF"]
    all_feat_rows = []

    for idx, row in df.iterrows():
        row_features = {
            'epoch': row['epoch'],
            'subject_id': row['subject_id'],
            'age': row['age'],
            'gender': row['gender'],
            'epilepsy': row['epilepsy']
        }
        for ch in channel_names:
            signal = row[ch]
            if isinstance(signal, np.ndarray):
                feats = compute_features(signal)
                ch_prefix = ch.replace(" ", "_")
                for k, v in feats.items():
                    row_features[f"{ch_prefix}_{k}"] = v

        all_feat_rows.append(row_features)

    features_df = pd.DataFrame(all_feat_rows)
    features_df.to_csv(output_csv, index=False)
    print(f"✅ Extracted features for {len(features_df)} segments → saved to {output_csv}")
    return features_df

if __name__ == "__main__":
    df = pd.read_pickle("preprocessed_data_updated.pkl")
    features_df = extract_all_features(df)
