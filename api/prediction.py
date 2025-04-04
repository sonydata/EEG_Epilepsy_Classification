import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from preprocessing import preprocess_eeg_file
from preprocessing_2dcnn import convert_epoch_to_spectrogram
import random
random.seed(42)

def aggregate_predictions(spectrogram_list, model, threshold=0.5):

    # Convert each spectrogram to channels-last format.
    X = np.array([np.transpose(s, (1, 2, 0)) for s in spectrogram_list])
    print(f'---Aggregating predictions from {len(spectrogram_list)} segments---')
    preds = model.predict(X)
    mean_prob = np.mean(preds[:, 1])
    final_label = 1 if mean_prob >= threshold else 0
    return final_label, mean_prob

def predict_eeg_recording(edf_path, model, threshold=0.5):
    #Process the edf file
    preprocessed_df = preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=2)
    
    if preprocessed_df is None or preprocessed_df.empty:
        raise ValueError("EEG file could not be preprocessed or no valid segments found.")
    
    channels = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF"]
    # Convert each 5-second segment (each row) into a spectrogram.
    spectrogram_list = preprocessed_df.apply(
        lambda row: convert_epoch_to_spectrogram(row, channels, fs=250, nperseg=128, noverlap=64), axis=1
    ).tolist()
    
    return aggregate_predictions(spectrogram_list, model, threshold)

