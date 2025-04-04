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
    # For relevant channel criteria check documentation
    '''“EEG FP1-REF” for the left frontal pole

        “EEG FP2-REF” for the right frontal pole

        “EEG F3-REF” for the left frontal region

        “EEG F4-REF” for the right frontal region

        “EEG C3-REF” for the left central region'''
        
    desired = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF"]
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

def preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=2):

    # 1. Charger le fichier EDF avec MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Resample (to 250 because it's the lowest sampling rate )
    raw.resample(250, verbose=False)
    
    # Filtrage passe-bande (1-45 Hz)
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
    
    # Skip EEGs less than 5s
    if raw.times[-1] < 5:
        print(f"Skipping {edf_path}: duration ({raw.times[-1]:.2f} s) is less than required 5s.")
        return None
    
    # Suppression des canaux non EEG
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_channels, verbose=False)
    
    # Selectionner les channels pertinents (channel selection from EDA ?)
    print(raw.ch_names)
    raw = select_relevant_channels(raw)
    if raw is None:
        return None
    
    # Segmentation
    epochs = mne.make_fixed_length_epochs(raw, duration=segment_lenght, preload=False, overlap=overlap, verbose=False)

    # Transform to dataframe and standadize

    df = epochs.to_data_frame() # epochs is returned by preprocess_eeg_file()
    df_std = standardize_dataframe(df.drop(['time','epoch', 'condition'], axis=1))
    result = pd.concat([df[['time','epoch']], df_std], axis=1)
    
    return collapse_epoch_df_by_channel(result)



