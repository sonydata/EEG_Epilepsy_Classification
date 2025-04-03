import os
import mne
import numpy as np
import pandas as pd

def standardize_dataframe(df):
    # Make a copy to avoid modifying the original dataframe
    df_standardized = df.copy()
    
    # Only standardize numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    
    for column in numeric_columns:
        mean = df[column].mean()
        std = df[column].std()
        df_standardized[column] = (df[column] - mean) / std
    
    return df_standardized

def select_relevant_channels(raw):
    desired = [
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
    #check if all desired channels are present; if not, skip this file
    if not all(ch in raw.ch_names for ch in desired):
        print("Skipping file because it doesn't have the full set of desired channels.")
        return None
    raw.pick_channels(desired, verbose=False)
    return raw


def collapse_epoch_df_by_channel(epoch_df):
    # Identify channel columns (exclude time, epoch, condition)
    channel_cols = [col for col in epoch_df.columns if col not in ['time', 'epoch']]
    # Group by epoch
    grouped = epoch_df.groupby('epoch')
    rows = []
    for epoch_num, group in grouped:
        group_sorted = group.sort_values('time')
        # For each channel, extract the 1D array for this epoch
        row = {'epoch': epoch_num}
        for ch in channel_cols:
            row[ch] = group_sorted[ch].values  # 1D array of length = number of time samples in the epoch
        rows.append(row)
    return pd.DataFrame(rows)

def preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=0.5):

    # Load raw EEG file 
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Resample to 250Hz (to match other files and ensure uniform time resolution)
    raw.resample(250, verbose=False)
    
    # Apply bandpass filter (1 - 45 Hz) to remove noise and artifacts
    raw.filter(1.0, 45.0, method='iir', verbose=False)

    # Keep only EEG channels (exclude EOG, ECG, etc.)
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_channels, verbose=False)
    
    # Select only the relevant subset of EEG channels needed for analysis
    raw = select_relevant_channels(raw)
    if raw is None:
        return None
    
    # Check if the total recording is long enough for at least one 5-second segment
    total_duration = raw.times[-1]
    if total_duration < segment_lenght:
        print(f"Skipping {edf_path}: total duration ({total_duration:.2f}s) is less than {segment_lenght}s.")
        return None
    
    # Segment the EEG into fixed-length (5-second) epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=segment_lenght, preload=True, overlap=overlap, verbose=False)

    # Ensure that at least one epoch was created
    if len(epochs) == 0:
        print(f"Skipping {edf_path}: no valid 5-second segments.")
        return None
    
    # Transform to dataframe (samples over time) and standadize signal values (z-score normalization)
    df = epochs.to_data_frame() # epochs is returned by preprocess_eeg_file()
    df_std = standardize_dataframe(df.drop(['time','epoch', 'condition'], axis=1))
    result = pd.concat([df[['time','epoch']], df_std], axis=1)
    
    return collapse_epoch_df_by_channel(result)

def flatten_df(processed_df):

    list_dfs = []
    # Loop through each row of the processed dataframe
    for idx, row in processed_df.iterrows():
        # Copy the epoch dataframe to avoid SettingWithCopyWarning
        epoch_df = pd.DataFrame(row['eeg_segments']).copy() 
        # Add metadata columns
        epoch_df['epilepsy'] = row['epilepsy']
        epoch_df['age'] = row['age']
        epoch_df['gender'] = row['gender']
        epoch_df['edf_path'] = row['edf_path']
        epoch_df['subject_id'] = row['subject_id']
        list_dfs.append(epoch_df)
    
    # Concatenate all epoch dataframes into a single dataframe
    final_df = pd.concat(list_dfs, ignore_index=True)
    return final_df


def preprocess(metadata, num_samples=100): #100 samples of EEG per class 
    
    # Keep only necessary columns from metadata
    df_filtered = metadata[['patient_group', 'age', 'gender', 'edf_path', 'subject_id']]
        
    # Map patient_group to a binary label: 1 for epilepsy, 0 for non-epilepsy
    df_filtered['epilepsy'] = df_filtered['patient_group'].map({'epilepsy': 1, 'no_epilepsy': 0})
    
    df_filtered = df_filtered.drop(columns=['patient_group'])
    
    df_sampled = df_filtered.groupby('epilepsy', group_keys=False).apply(
        lambda x: x.sample(n=min(num_samples, len(x)), random_state=42))
    
    print(f'Remaining samples: {len(df_sampled)}')
    
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file) #apply preprocessing to each eeg
    
    df_sampled = df_sampled[df_sampled['eeg_segments'].notnull()] #drop rows where egg_segments is empty 
    
    print('Preprocessing done')
    print(f'Remaining samples: {len(df_sampled)}')
    
    # Flatten
    final_df = flatten_df(df_sampled)

    # Save result to pickle
    final_df.to_pickle('preprocessed_data_updated.pkl') 
    
    return final_df

if __name__ == "__main__":
    metadata_df = pd.read_excel('eeg_metadata.xlsx') #metadata df obtained from previous extraction

    processed_df = preprocess(metadata_df)

    print(processed_df)