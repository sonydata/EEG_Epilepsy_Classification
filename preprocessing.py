import os
import mne
import numpy as np
import pandas as pd


def preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=0):

    # 1. Charger le fichier EDF avec MNE
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 2. Filtrage passe-bande (1-45 Hz)
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)

    # 3. Suppression des canaux non EEG
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_channels)
    
    # Selectionner les channels pertinents (channel selection from EDA ?)
    
    
    #  4. Normalisation canal par canal (centrage-réduction)
    data = raw.get_data()
    data_norm = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    raw._data = data_norm

    # 5. Segmentation
    epochs = mne.make_fixed_length_epochs(raw, duration=segment_lenght, preload=False, overlap=overlap)

    # 6. Transform to np array ?

    return epochs


def preprocess(metadata):
    df_filtered = metadata[metadata['montage'] == '01_tcp_ar'] #select only average reference montage 
    
    #DROP COLUMNS
    
    df_sampled = df_filtered.groupby('patient_group', group_keys=False).apply(lambda x: x.sample(n=2, random_state=42)) #sample balanced classes
    print(f'Remaining samples: {len(df_sampled)}')
    
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file) #apply preprocessing to each eeg
    
    print('Preprocessing done')
    print(f'Remaining samples: {len(df_sampled)}')

    return df_sampled

metadata_df = pd.read_excel('eeg_metadata.xlsx') #metadata df obtained from previous extraction

processed_df = preprocess(metadata_df)

print(processed_df)