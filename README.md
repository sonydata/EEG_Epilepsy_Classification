# EEG_Epilepsy_Classification
Final group project aiming to identify epilepsy from EEG signals using machine learning models 

**Dataset:  TUH EEG Epilepsy Corpus v2.0.1**

Subjects were sorted into epilepsy and no epilepsy categories by searching
the associated EEG reports for indications as to an epilepsy/no epilepsy 
diagnosis based on clinical history, medications at the time of recording, 
and EEG features associated with epilepsy such as spike and sharp waves.
A board-certified neurologist, Daniel Goldenholz, and his research team
reviewed and verified the decisions about each patient.

BASIC STATISTICS:
```
  |-------------------------------------------------------|
  | Description |  Epilepsy   | No Epilepsy |    Total    |
  |-------------+-------------+-------------+-------------|
  | Patients    |         100 |         100 |         200 |
  |-------------+-------------+-------------+-------------|
  | Sessions    |         530 |         168 |         698 |
  |-------------+-------------+-------------+-------------|
  | Files       |       1,785 |         513 |       2,298 |
  |-------------------------------------------------------|
```
Reference: Veloso, L., McHugh, J. R., von Weltin, E., Obeid, I., & Picone,
 J. (2017). Big Data Resources for EEGs: Enabling Deep Learning
 Research. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE
 Signal Processing in Medicine and Biology Symposium
 (p. 1). Philadelphia, Pennsylvania, USA: IEEE.

**Preprocessing pipeline**
```
📄 eeg_metadata.xlsx
│
├── Each row: one EEG session (.edf file)
│
▼
📋 Sample metadata (100 per class)
│
├── epilepsy (1)     ─────┐
└── no_epilepsy (0)  ─────┘
     │
     ▼
📂 Load .edf file using MNE
     │
     ├── Resample to 250 Hz
     ├── Bandpass filter (1–45 Hz)
     ├── Select EEG-only channels
     ├── Select subset of relevant channels (F7, T3, T4, T6, CZ)
     ├── Skip if:
     │   ├── Duration < 5 sec
     │   └── Missing channels
     │
     ▼
🧠 Segment into overlapping 5-second epochs
     │
     ├── Drop if no valid epochs
     ▼
📊 Convert to long-format DataFrame
     │
     ├── Z-score normalize each channel
     ▼
📦 Collapse into one row per 5-sec epoch
     (arrays per channel)
     ▼
📄 Add to 'eeg_segments' column
     ▼
📋 Merge with metadata (age, gender, label...)
     ▼
🧾 Flatten: each row = one 5-second segment
     ▼
💾 Save as Pickle → `preprocessed_data_updated.pkl`
```
