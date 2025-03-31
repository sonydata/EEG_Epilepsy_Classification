# EEG_Epilepsy_Classification
Final group project aiming to identify epilepsy from EEG signals using machine learning models 

Dataset:  TUH EEG Epilepsy Corpus
Version: v2.0.1

Subjects were sorted into epilepsy and no epilepsy categories by searching
the associated EEG reports for indications as to an epilepsy/no epilepsy 
diagnosis based on clinical history, medications at the time of recording, 
and EEG features associated with epilepsy such as spike and sharp waves.
A board-certified neurologist, Daniel Goldenholz, and his research team
reviewed and verified the decisions about each patient.

Reference: Veloso, L., McHugh, J. R., von Weltin, E., Obeid, I., & Picone,
 J. (2017). Big Data Resources for EEGs: Enabling Deep Learning
 Research. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE
 Signal Processing in Medicine and Biology Symposium
 (p. 1). Philadelphia, Pennsylvania, USA: IEEE.

The files subject_ids_*.list contain unique IDs for each subject that is
part of the corpus. We also provide links to the edf data for users who
have not downloaded the entire corpus.

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
