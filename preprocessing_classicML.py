# EEG Preprocessing Pipeline for Channel Selection & ICA-Based Artifact Removal
# ---------------------------------------------------------------
# This script preprocesses EEG data from the TUH EEG Corpus for patient-level classification.
# It performs the following steps:
# 
# - Filters EEG recordings to a consistent 5-channel subset (P3, T3, C3, CZ, C4)
# - Removes noise (bandpass: 0.5–45 Hz, notch: 60 Hz)
# - Applies ICA to remove artifacts (e.g., muscle/eye noise), preserving all EEG channels (n_components=5)
# - Segments EEG into 5-second overlapping epochs (50% overlap)
# - Standardizes data and collapses each segment to one row (1D vector per channel)
# - Samples up to 20 patients per class (epileptic vs non-epileptic)
# - Keeps a maximum of 150 segments per patient to preserve diversity
# 
# Final output: `preprocessed_data_updated.pkl`, a flattened DataFrame with segment-level EEG features + metadata
# Note: Minimum number of usable 5-second segments per patient depends on EEG length (≥30s required); with 50% overlap, a 30s recording yields 11 segments, a 60s recording yields ~21.

import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from mne.preprocessing import ICA

# ------------------------- Utility Functions -------------------------

def standardize_dataframe(df):
    # Standardizes only numeric columns (z-score)
    numeric_cols = df.select_dtypes(include=np.number).columns
    return (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

def select_relevant_channels(raw, desired=None):
    # Use a consistent set of standard motor/parietal channels for cross-subject comparability
    desired = ["EEG P3-REF", "EEG T3-REF", "EEG C3-REF", "EEG CZ-REF", "EEG C4-REF"]
    if not all(ch in raw.ch_names for ch in desired):
        return None
    raw.pick_channels(desired, verbose=False)
    raw.reorder_channels(desired)  # Reorder channels to consistent order across files
    return raw

def apply_ica_artifact_removal(raw, n_components=5, random_state=97):
    ica = ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    ica.fit(raw)

    # Get component activations (sources)
    sources = ica.get_sources(raw).get_data()

    # Compute kurtosis for each component
    kurtosis_scores = kurtosis(sources, axis=1, fisher=False)

    # Auto-exclude top 10% most non-Gaussian components (optional tuning)
    threshold = np.percentile(kurtosis_scores, 90)
    to_exclude = np.where(kurtosis_scores > threshold)[0].tolist()
    ica.exclude = to_exclude

    print(f"ICA: Marking {len(to_exclude)} component(s) for exclusion: {to_exclude}")

    return ica, ica.apply(raw.copy(), exclude=ica.exclude)

def collapse_epoch_df_by_channel(epoch_df):
    # Collapse epoch DataFrame into one row per segment with 1D array per channel
    grouped = epoch_df.groupby('epoch')
    return pd.DataFrame([
        {'epoch': epoch, **{ch: group.sort_values('time')[ch].values for ch in group.columns if ch not in ['time', 'epoch']}}
        for epoch, group in grouped
    ])

def preprocess_eeg_file(edf_path, fmin=0.5, fmax=45.0, segment_length=5, overlap=0.5):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None

    raw = select_relevant_channels(raw)
    if raw is None:
        return None

    duration = raw.times[-1]
    # Skip sessions that are too short (<30s) or too long (>1 hour), often long-term monitoring
    if duration > 3600 or duration < 30:
        return None

    raw.resample(250, verbose=False)  # Normalize sampling rate
    raw.notch_filter(freqs=60, verbose=False)  # Remove power line noise
    raw.filter(fmin, fmax, method='iir', verbose=False)  # Bandpass filter to focus on neural frequencies

    _, raw = apply_ica_artifact_removal(raw)  # Apply ICA and replace raw with cleaned version

    epochs = mne.make_fixed_length_epochs(raw, duration=segment_length, overlap=overlap, preload=True, verbose=False)
    if len(epochs) == 0:
        return None

    df = epochs.to_data_frame()
    df_std = standardize_dataframe(df.drop(columns=['time', 'epoch', 'condition']))
    return collapse_epoch_df_by_channel(pd.concat([df[['time', 'epoch']], df_std], axis=1))

# ------------------------- Flatten Function -------------------------

def flatten_df(df, max_segments=150): #set max to 150 segments of 5sec total to have more signal diversity
    # Flatten patient-wise EEG segments into a single DataFrame with metadata
    all_patients = []
    for subject_id, group in df.groupby('subject_id'):
        segments = pd.concat([seg for seg in group['eeg_segments'] if isinstance(seg, pd.DataFrame)], ignore_index=True)
        if segments.empty:
            continue
        sampled = segments.sample(n=min(max_segments, len(segments)), random_state=42)
        for col in ['epilepsy', 'age', 'gender']:
            sampled[col] = group[col].iloc[0]
        sampled['subject_id'] = subject_id
        all_patients.append(sampled)

    return pd.concat(all_patients, ignore_index=True) if all_patients else pd.DataFrame()

# ------------------------- Main Preprocessing -------------------------

def preprocess(metadata, patients_per_class=10, segments_per_patient=20):
    # Clean and relabel metadata
    df = metadata[['patient_group', 'age', 'gender', 'edf_path', 'subject_id']].copy()
    df['epilepsy'] = df['patient_group'].map({'epilepsy': 1, 'no_epilepsy': 0})
    df.drop(columns='patient_group', inplace=True)

    # Randomly sample up to N patients per class (balanced subset)
    sampled_patients = df.drop_duplicates('subject_id').groupby('epilepsy', group_keys=False).apply(
        lambda x: x.sample(n=min(patients_per_class, len(x)), random_state=42)
    )
    sampled_ids = sampled_patients['subject_id']
    df_sampled = df[df['subject_id'].isin(sampled_ids)]

    # Preprocess EDFs and extract EEG segments
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file)
    df_sampled = df_sampled[df_sampled['eeg_segments'].notnull()]  # Drop failed extractions

    # Flatten into one segment per row
    final_df = flatten_df(df_sampled, max_segments=segments_per_patient)
    final_df.to_pickle('preprocessed_data_updated.pkl')

    print(f"✅ Preprocessing done. Patients: {final_df['subject_id'].nunique()}, Segments: {len(final_df)}")
    return final_df

# ------------------------- Entry Point -------------------------

if __name__ == "__main__":
    metadata_df = pd.read_excel('eeg_metadata.xlsx')
    processed_df = preprocess(metadata_df, patients_per_class=10, segments_per_patient=20)
