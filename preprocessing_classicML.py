import os
import mne
import numpy as np
import pandas as pd

def standardize_dataframe(df):
    df_standardized = df.copy()
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        mean = df[column].mean()
        std = df[column].std()
        df_standardized[column] = (df[column] - mean) / std
    return df_standardized

def select_relevant_channels(raw):
    desired = ["EEG F7-REF", "EEG T3-REF", "EEG T4-REF", "EEG T6-REF", "EEG CZ-REF"]
    if not all(ch in raw.ch_names for ch in desired):
        print("⛔️ Skipping file: missing required channels.")
        return None
    raw.pick_channels(desired, verbose=False)
    return raw

def collapse_epoch_df_by_channel(epoch_df):
    channel_cols = [col for col in epoch_df.columns if col not in ['time', 'epoch']]
    grouped = epoch_df.groupby('epoch')
    rows = []
    for epoch_num, group in grouped:
        group_sorted = group.sort_values('time')
        row = {'epoch': epoch_num}
        for ch in channel_cols:
            row[ch] = group_sorted[ch].values
        rows.append(row)
    print(f"✅ collapse_epoch_df_by_channel: returned {len(rows)} segments")
    return pd.DataFrame(rows)

def preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_length=5, overlap=0.5):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"❌ Failed to read {edf_path}: {e}")
        return None

    if raw.times[-1] > 3600:
        print(f"⏩ Skipping {edf_path}: session longer than 1 hour ({raw.times[-1]:.2f} s).")
        return None

    raw.resample(250, verbose=False)
    raw.filter(fmin, fmax, method='iir', verbose=False)

    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_channels, verbose=False)

    raw = select_relevant_channels(raw)
    if raw is None:
        return None

    if raw.times[-1] < segment_length:
        print(f"⏩ Skipping {edf_path}: duration too short ({raw.times[-1]:.2f}s).")
        return None

    epochs = mne.make_fixed_length_epochs(raw, duration=segment_length, preload=True, overlap=overlap, verbose=False)
    print(f"📏 {edf_path} → {len(epochs)} raw epochs")

    if len(epochs) == 0:
        print(f"⚠️ No epochs found in {edf_path}")
        return None

    df = epochs.to_data_frame()
    df_std = standardize_dataframe(df.drop(['time', 'epoch', 'condition'], axis=1))
    result = pd.concat([df[['time', 'epoch']], df_std], axis=1)

    collapsed = collapse_epoch_df_by_channel(result)

    return collapsed if not collapsed.empty else None

def flatten_df(processed_df, max_segments_per_patient=20):
    list_dfs = []

    # 🔁 Group all EEGs by patient
    grouped = processed_df.groupby('subject_id')

    for subject_id, group in grouped:
        all_segments = []

        for _, row in group.iterrows():
            segment_df = row['eeg_segments']
            if isinstance(segment_df, pd.DataFrame) and not segment_df.empty:
                all_segments.append(segment_df)

        if not all_segments:
            continue

        patient_df = pd.concat(all_segments, ignore_index=True)

        # ✅ Sample max segments per patient across all EEGs
        if len(patient_df) > max_segments_per_patient:
            patient_df = patient_df.sample(n=max_segments_per_patient, random_state=42)

        patient_df['epilepsy'] = group['epilepsy'].iloc[0]
        patient_df['age'] = group['age'].iloc[0]
        patient_df['gender'] = group['gender'].iloc[0]
        patient_df['subject_id'] = subject_id

        list_dfs.append(patient_df)

    if not list_dfs:
        print("⚠ No data was added to the final list!")
        return pd.DataFrame()

    flat_df = pd.concat(list_dfs, ignore_index=True)
    print(f"\n✅ Final number of segments: {len(flat_df)}")
    print(f"🧾 Patients included: {flat_df['subject_id'].nunique()}")
    return flat_df

def preprocess(metadata, num_patients_per_class=5, max_segments_per_patient=20):
    df_filtered = metadata[['patient_group', 'age', 'gender', 'edf_path', 'subject_id']].copy()
    df_filtered['epilepsy'] = df_filtered['patient_group'].map({'epilepsy': 1, 'no_epilepsy': 0})
    df_filtered = df_filtered.drop(columns=['patient_group'])

    # 🔁 Step 1: Sample patients per class (not EDFs!)
    unique_patients = df_filtered.drop_duplicates('subject_id')
    sampled_patients = unique_patients.groupby('epilepsy', group_keys=False).apply(
        lambda x: x.sample(n=min(num_patients_per_class, len(x)), random_state=42)
    )

    print("✅ Unique patients sampled per class:", sampled_patients['epilepsy'].value_counts())

    # 🔁 Step 2: Get all EEGs for those sampled patients
    df_sampled = df_filtered[df_filtered['subject_id'].isin(sampled_patients['subject_id'])]
    print(f"📦 Total EEG files to process: {len(df_sampled)} from {df_sampled['subject_id'].nunique()} patients")

    # Preprocess each EEG file
    df_sampled['eeg_segments'] = df_sampled['edf_path'].apply(preprocess_eeg_file)
    df_sampled = df_sampled[df_sampled['eeg_segments'].notnull()]

    print("👥 Unique patients after preprocessing:", df_sampled['subject_id'].nunique())
    print(f"🧠 Remaining usable EEGs: {len(df_sampled)}")

    final_df = flatten_df(df_sampled, max_segments_per_patient=max_segments_per_patient)
    final_df.to_pickle('preprocessed_data_updated.pkl')
    return final_df

# ============== Entry Point ==============
if __name__ == "__main__":
    metadata_df = pd.read_excel('eeg_metadata.xlsx')
    processed_df = preprocess(metadata_df)
    print(processed_df.head())
