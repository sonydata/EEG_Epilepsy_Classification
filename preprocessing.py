import mne
import numpy as np
import pandas as pd


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
    raw.pick_channels(desired)
    return raw

def preprocess_eeg_file(edf_path, max_duration=30,fmin=1.0, fmax=45.0, segment_lenght=5, overlap=0):

    # 1. Charger le fichier EDF avec MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Skip EEGs less than max duration (in seconds)
    if raw.times[-1] < max_duration:
        print(f"Skipping {edf_path}: duration ({raw.times[-1]:.2f} s) is less than required {max_duration} s.")
        return None
    
    # Crop the EEG to get the same duration
    raw.crop(tmin=0, tmax=max_duration)
    
    # Resample (to 250 because it's the lowest sampling rate )
    raw.resample(250, verbose=False)
    
    # 2. Filtrage passe-bande (1-45 Hz)
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)

    # 3. Suppression des canaux non EEG
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_channels)
    
    # Selectionner les channels pertinents (channel selection from EDA ?)
    print(raw.ch_names)
    raw = select_relevant_channels(raw)
    if raw is None:
        return None
    
    #  4. Normalisation canal par canal (centrage-réduction)
    data = raw.get_data()
    data_norm = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    raw._data = data_norm

    # 5. Segmentation
    epochs = mne.make_fixed_length_epochs(raw, duration=segment_lenght, preload=False, overlap=overlap)

    # Transform to np array and converts the 3D array to a nested Python list 
    return epochs.get_data().tolist()


def preprocess(metadata):
    df_filtered = metadata[metadata['montage'] == '01_tcp_ar'] #select only average reference montage 
    
    # DROP COLUMNS: Keep only patient_group, age, gender, and edf_path
    df_filtered = df_filtered[['patient_group', 'age', 'gender', 'edf_path']]
    
    # Map patient_group to a binary label: 1 for epilepsy, 0 for non-epilepsy
    df_filtered['epilepsy'] = df_filtered['patient_group'].map({'epilepsy': 1, 'no_epilepsy': 0})
    
    df_filtered = df_filtered.drop(columns=['patient_group'])
    
    
    df_sampled = df_filtered.groupby('epilepsy', group_keys=False).apply(lambda x: x.sample(n=5, random_state=42)) #sample balanced classes
    print(f'Remaining samples: {len(df_sampled)}')
    
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file) #apply preprocessing to each eeg

    print('Preprocessing done')
    print(f'Remaining samples: {len(df_sampled)}')

    return df_sampled

metadata_df = pd.read_excel('eeg_metadata.xlsx') #metadata df obtained from previous extraction

processed_df = preprocess(metadata_df)

print(processed_df)
#print(processed_df['eeg_segments'].apply(lambda x: x.size if x is not None else 0)) #check array size
print(processed_df['eeg_segments'].apply(lambda x: len(x) if x is not None else 0))

processed_df.to_csv("processed_eeg_data.csv")