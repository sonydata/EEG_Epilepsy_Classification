import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy import sample_entropy, perm_entropy
import pywt #pip install PyWavelets

# EEG frequency bands of interest for epilepsy detection 
BANDS = {
    'delta': (0.5, 4),     # Associated with deep sleep or brain dysfunction
    'theta': (4, 8),       # Seen in drowsiness, meditation, or pathology
    'alpha': (8, 12),      # Dominant in relaxed wakefulness, suppressed during seizures
    'beta': (12, 30),      # Linked to alertness and sometimes pathological activity
    'gamma': (30, 45)      # High-frequency activity, occasionally linked to seizure onset
}

# Parameters for Wavelet Transform
WAVELET = 'db4'         # Common wavelet used in EEG analysis for time-frequency decomposition
DWT_LEVEL = 4           # Level of decomposition

def bandpower(data, sf, band, relative=True):
    """
    Computes the power of a signal in a given frequency band using Welch's method.
    If relative=True, returns the proportion of power in that band relative to total power.
    """
    low, high = band
    nperseg = min(256, len(data))  # Number of samples per segment for Welch
    freqs, psd = welch(data, sf, nperseg=nperseg)
    idx = np.logical_and(freqs >= low, freqs <= high)
    power = np.trapz(psd[idx], freqs[idx])
    if relative:
        total = np.trapz(psd, freqs)
        power = power / total if total > 0 else 0
    return power


def compute_features(signal, sf=250):
    """
    Extracts multiple types of features from a single-channel EEG segment:
    - Time-domain statistics
    - Frequency-band relative powers
    - Entropy/complexity metrics
    - Wavelet-based features
    """
    features = {}

    # --- Time-domain features ---
    features['mean'] = np.mean(signal)
    features['var'] = np.var(signal)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal, fisher=False)
    features['zcr'] = np.mean(np.diff(np.sign(signal)) != 0)  # Zero-crossing rate
    features['tkeo'] = np.mean(signal[1:-1]**2 - signal[:-2]*signal[2:]) if len(signal) > 2 else 0  # Teager-Kaiser Energy Operator

    # --- Frequency-domain features ---
    for name, (lo, hi) in BANDS.items():
        features[f'{name}_power'] = bandpower(signal, sf, (lo, hi))

    # --- Entropy/complexity features ---
    features['samp_entropy'] = sample_entropy(signal, 2, 0.2 * np.std(signal))  # Sample Entropy
    features['perm_entropy'] = perm_entropy(signal, order=3, normalize=True)   # Permutation Entropy

    # --- Wavelet-based features ---
    coeffs = pywt.wavedec(signal, wavelet=WAVELET, level=DWT_LEVEL)
    for i, c in enumerate(coeffs):
        band = f"A{DWT_LEVEL}" if i == 0 else f"D{DWT_LEVEL - i + 1}"
        energy = np.sum(c**2)
        var = np.var(c)
        p = np.abs(c)**2
        p /= (np.sum(p) + 1e-12)  # Normalize to avoid division by 0
        shannon = -np.sum(p * np.log2(p + 1e-12))  # Shannon entropy
        features[f'{band}_energy'] = energy
        features[f'{band}_var'] = var
        features[f'{band}_entropy'] = shannon

    return features


def extract_all_features(df, output_csv='eeg_features_updated.csv'):
    """
    Loops through the preprocessed EEG DataFrame:
    - Applies feature extraction per epoch 
    - Merges extracted features with patient metadata
    - Saves full feature set to CSV
    """
    # Channels used during preprocessing
    channel_names = [
    #"EEG C3-REF",  # Left central - Sensorimotor cortex involvement
    #"EEG C4-REF",  # Right central - Contralateral propagation detection
    #"EEG P3-REF",  # Left parietal - Posterior seizure detection
    #"EEG P4-REF",  # Right parietal - Low artifact contamination
    "EEG F7-REF",  # Left frontal - Detects frontal lobe epilepsy patterns
    #"EEG F8-REF",  # Right frontal - Supplementary motor area seizures
    "EEG T3-REF",  # Left temporal lobe - Most critical for focal seizures
    "EEG T4-REF",  # Right temporal lobe - Second most common seizure origin
    #"EEG T5-REF",  # Left posterior temporal - Temporal-parietal junction activity
    "EEG T6-REF",  # Right posterior temporal - Interictal spike detection
    "EEG CZ-REF"  # Central vertex - Essential for generalized seizure patterns
]
    all_feat_rows = []

    for idx, row in df.iterrows():
        epoch_features = {
        'epoch': row['epoch'],
        'subject_id': row['subject_id'],
        'age': row['age'],
        'gender': row['gender'],
        'epilepsy': row['epilepsy']
    }
        
        for ch in channel_names:
            signal = row[ch]
            if signal is None:
                continue

            features = compute_features(signal)

            # Prefix each feature with the channel name to avoid collisions
            ch_prefix = ch.replace(" ", "_")  # e.g., "EEG FP1-REF" -> "EEG_FP1-REF"
            for feat_name, value in features.items():
                epoch_features[f"{ch_prefix}_{feat_name}"] = value
            
        all_feat_rows.append(epoch_features)

    # Final DataFrame and export
    feature_df = pd.DataFrame(all_feat_rows)
    feature_df.to_csv(output_csv, index=False)
    print(f"Extracted features for {len(feature_df)} rows → saved to {output_csv}")
    return feature_df


if __name__ == "__main__":
    # Load the preprocessed EEG data where each row contains a stringified NumPy array
    df = pd.read_pickle("preprocessed_data_updated.pkl")
    features_df = extract_all_features(df)
