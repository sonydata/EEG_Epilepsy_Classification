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

def preprocess_eeg_file(edf_path, max_duration=30,fmin=1.0, fmax=45.0, segment_lenght=5, overlap=0):

    # 1. Charger le fichier EDF avec MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Skip EEGs less than max duration (in seconds)
    if raw.times[-1] < max_duration:
        print(f"Skipping {edf_path}: duration ({raw.times[-1]:.2f} s) is less than required {max_duration} s.")
        return None
    
    # Crop the EEG to get the same duration
    raw.crop(tmin=0, tmax=max_duration, verbose=False)
    
    # Resample (to 250 because it's the lowest sampling rate )
    raw.resample(250, verbose=False)
    
    # Filtrage passe-bande (1-45 Hz)
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)

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

def flatten_df(processed_df):

    list_dfs = []
    # Loop through each row of the processed dataframe
    for idx, row in processed_df.iterrows():
        # Copy the epoch dataframe to avoid SettingWithCopyWarning
        #epoch_df = row['eeg_segments'].copy()
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


def preprocess(metadata, num_samples=2):
    df_filtered = metadata[metadata['montage'] == '01_tcp_ar'] #select only average reference montage 
    
    # DROP COLUMNS: Keep only patient_group, age, gender, and edf_path
    df_filtered = df_filtered[['patient_group', 'age', 'gender', 'edf_path', 'subject_id']]
    
    # Map patient_group to a binary label: 1 for epilepsy, 0 for non-epilepsy
    df_filtered['epilepsy'] = df_filtered['patient_group'].map({'epilepsy': 1, 'no_epilepsy': 0})
    
    df_filtered = df_filtered.drop(columns=['patient_group'])
    
    
    df_sampled = df_filtered.groupby('epilepsy', group_keys=False).apply(
        lambda x: x.sample(n=min(num_samples, len(x)), random_state=42)) #samples either num_samples (from each group) or total number of samples if it's smaller that num_samples
    #example num_samples = 50 but there are only 35 epilepsy EEGS
    #returns 35 samples of epilepsy EEGs and 50 of non epilepsy EEgs
    
    print(f'Remaining samples: {len(df_sampled)}') #example 85 = 35 + 50 
    
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file) #apply preprocessing to each eeg
    
    df_sampled = df_sampled[df_sampled['eeg_segments'].notnull()] #drop rows where egg_segments is empty 
    #(reasons EDF file does not contain desired channels or duration is less than 30 sec)
    
    print('Preprocessing done')
    print(f'Remaining samples: {len(df_sampled)}')
    
    #Flatten
    final_df = flatten_df(df_sampled)

    #Save result to excel
    #final_df.to_excel('preprocessed_data.xlsx')
    final_df.to_pickle('preprocessed_data_updated.pkl') #this version works better w/ numpy arrays
    
    return final_df

if __name__ == "__main__":
    metadata_df = pd.read_excel('eeg_metadata.xlsx') #metadata df obtained from previous extraction

    processed_df = preprocess(metadata_df)

    print(processed_df)

    print(processed_df['EEG FP1-REF'].apply(lambda x: x.size))