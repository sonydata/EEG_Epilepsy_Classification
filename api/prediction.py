import numpy as np
import torch
import tensorflow as tf
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from preprocessing import preprocess_eeg_file
from preprocessing_2dcnn import convert_epoch_to_spectrogram
from preprocessing_epilepsynet import *
from EpilepsyNet_model import TimeSeriesAttentionClassifier
from eegnet_model import EEGNet



def aggregate_predictions(spectrogram_list, model, threshold=0.5):

    # Convert each spectrogram to channels-last format.
    X = np.array([np.transpose(s, (1, 2, 0)) for s in spectrogram_list])
    print(f'---Aggregating predictions from {len(spectrogram_list)} segments---')
    preds = model.predict(X)
    mean_prob = np.mean(preds[:, 1])
    final_label = 1 if mean_prob >= threshold else 0
    return final_label, mean_prob

def predict_eeg_recording(edf_path, model_name='2DCNN', threshold=0.5):
    
    
    if model_name == '2DCNN':
        
        model = tf.keras.models.load_model('model1_2dcnn.h5')
        channels = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF"]
        #Process the edf file
        preprocessed_df = preprocess_eeg_file(edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=2,desired=channels)
    
        if preprocessed_df is None or preprocessed_df.empty:
            raise ValueError("EEG file could not be preprocessed or no valid segments found.")
        
        # Convert each 5-second segment (each row) into a spectrogram.
        spectrogram_list = preprocessed_df.apply(
            lambda row: convert_epoch_to_spectrogram(row, channels, fs=250, nperseg=128, noverlap=64), axis=1
        ).tolist()
        
        return aggregate_predictions(spectrogram_list, model, threshold)
    
    elif model_name == 'EEGNet':
        loaded = joblib.load("eegnet_model.joblib")
        # Extract the actual state dictionary.
        state_dict = loaded["model_state_dict"]
        
        model = EEGNet(n_channels=21, n_samples=1250, num_classes=2)
        model.load_state_dict(state_dict)
        
        # Define the channels to use for EEGNet
        channels = [
                    'EEG FP1-REF',  # Left frontal pole
                    'EEG FP2-REF',  # Right frontal pole
                    'EEG F3-REF',   # Left frontal
                    'EEG F4-REF',   # Right frontal
                    'EEG C3-REF',   # Left central
                    'EEG C4-REF',   # Right central
                    'EEG P3-REF',   # Left parietal
                    'EEG P4-REF',   # Right parietal
                    'EEG O1-REF',   # Left occipital
                    'EEG O2-REF',   # Right occipital
                    'EEG F7-REF',   # Left lateral frontal
                    'EEG F8-REF',   # Right lateral frontal
                    'EEG T3-REF',   # Left temporal (anterior)
                    'EEG T4-REF',   # Right temporal (anterior)
                    'EEG T5-REF',   # Left temporal (posterior)
                    'EEG T6-REF',   # Right temporal (posterior)
                    'EEG FZ-REF',   # Frontal midline
                    'EEG CZ-REF',   # Central midline
                    'EEG PZ-REF',   # Parietal midline
                    'EEG ROC-REF',  # Right occipital (often used as reference or an extra site)
                    'EEG LOC-REF'   # Left occipital (often used as reference or an extra site)
                    ]
        # Process the edf file
        preprocessed_df = preprocess_eeg_file(
            edf_path, fmin=1.0, fmax=45.0, segment_lenght=5, overlap=0, desired=channels
        )
    
        if preprocessed_df is None or preprocessed_df.empty:
            raise ValueError("EEG file could not be preprocessed or no valid segments found.")
        
        # For EEGNet, we use the raw time series data directly.
        # Convert each 5-second segment (row) to a 2D timeseries array of shape (n_channels, n_samples)
        timeseries_list = preprocessed_df.apply(
            lambda row: convert_epoch_to_timeseries(row, channels), axis=1
        ).tolist()
        
        return aggregate_predictions_EEGNET(timeseries_list, model, threshold)
    
    elif model_name == 'EpilepsyNet':
        
        raw = mne.io.read_raw_edf(edf_path,
                                preload=True,
                                verbose='ERROR')
        
        eeg_cols = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 
                'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 
                'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 
                'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 
                'EEG T1', 'EEG T2', 'EEG FZ', 'EEG CZ',
                'EEG PZ']

        parameters = {
            'eeg_cols':eeg_cols,
            'segment_duration':60.0,        # 60 second segments
            'n_segments_per_file':12,       # Split into 12 epochs (5 sec each)
            'samples_per_segment':1250,     # 1250 samples per segment (250 Hz sampling rate)
            'random_state':42  
            }
        
        X = process_raw_files(
            raw_file=raw,
            eeg_cols=eeg_cols,
            segment_duration=parameters['segment_duration'],
            n_segments_per_file=parameters['n_segments_per_file'],
            random_state=parameters['random_state']
            )

        X_std = standardize_data(X)
        corr_matrix = compute_correlation_matrix(X_std)
        # print('Correlation matrix shape :',corr_matrix.shape)

        upper_triangle_matrix = extract_upper_triangle(corr_matrix)
        # print('Upper Triangle shape :',upper_triangle_matrix.shape)

        X_tensor = torch.tensor(upper_triangle_matrix, dtype=torch.float32)
        X_tensor = X_tensor.unsqueeze(0)
        
        # Model parameters
        input_dim = 210  # Size of flattened upper triangle (21*20/2)
        embed_dim = 256  # Embedding dimension
        num_heads = 16    # Number of attention heads7  
        
        model = TimeSeriesAttentionClassifier(input_dim, embed_dim, num_heads)
        model.load_state_dict(torch.load('EpilepsyNet.pth'))
        model.eval()
        print('¨'*50)
        print('Model Prediction :')

        outputs, _ = model(X_tensor)
        # For binary classification with sigmoid, prediction is 1 if output > 0.5
        predicted = (outputs >= 0.5).float()       
        
        return int(predicted), outputs.float().squeeze().item()


def convert_epoch_to_timeseries(epoch_row, channels):

    ts_list = []
    for ch in channels:
        # Check if the channel is in the epoch_row; if not, skip it.
        if ch in epoch_row:
            ts = epoch_row[ch]
            ts_list.append(ts)
    return np.stack(ts_list, axis=0)


def aggregate_predictions_EEGNET(segment_list, model, threshold):
    """
    Given a list of segments (raw time series data for EEGNet) and a trained
    PyTorch model, predict on each segment and then aggregate the predictions.
    
    For each segment, the model returns a probability vector.
    This function averages the predicted probabilities across segments,
    and then compares the average probability for class 1 with the provided threshold
    to decide the final predicted class.
    
    Parameters:
      segment_list : list of numpy arrays
          Each element is a 2D numpy array with shape (n_channels, n_samples)
          representing one EEG segment.
      model : a trained PyTorch model that accepts input of shape 
          (batch_size, n_channels, n_samples) and outputs probabilities (or logits) for each class.
      threshold : float
          The probability threshold to decide class 1.
    
    Returns:
      final_class : int
          The aggregated predicted class (0 or 1).
    """
    model.eval()
    preds = []
    
    with torch.no_grad():
        for seg in segment_list:
            # Convert the segment to a torch tensor (float32) 
            # Expected shape: (n_channels, n_samples)
            seg_tensor = torch.tensor(seg, dtype=torch.float32)
            # Add a batch dimension -> shape: (1, 1, n_channels, n_samples)
            seg_tensor = seg_tensor.unsqueeze(0).unsqueeze(0) 
            
            # Forward pass: get the model's output.
            # If your model returns logits, you may need to apply softmax.
            output = model(seg_tensor)
            
            # Check if the output is probabilities already or logits.
            # For safety, let's apply softmax to ensure we have probabilities.
            prob = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            preds.append(prob)
    
    # Average the predictions over all segments
    avg_pred = np.mean(preds, axis=0)
    # For binary classification assume avg_pred[1] is the probability for class 1.
    final_class = int(avg_pred[1] >= threshold)

    return final_class, avg_pred[1]



import numpy as np

def predict_ensemble_eeg_recording(edf_path, ensemble_method, threshold=0.5):
    """
    The function aggregates the probability scalars from each model and then:
      - For soft voting (method="average"): averages the probabilities
      - For hard voting (method="voting"): uses majority voting (each model votes 1 if 
        its probability is >= threshold, else 0)
    
    Parameters:
      edf_path : str
          Path to the EEG EDF file.
      ensemble_method : str
          Aggregation method, either "average" for soft voting or "voting" for hard voting.
      threshold : float, default=0.5
          The probability threshold to decide class 1.
    
    Returns:
      final_class : int
          The final aggregated predicted class (0 or 1).
      aggregated : float or list
          For "average", the average probability as a float;
          for "voting", the list of votes from each model.
    """
    # Lists to collect probabilities and votes.
    pred_prob_list = []
    votes = []
    
    # Iterate over the three model types.
    for model_name in ["2DCNN", "EEGNet", "EpilepsyNet"]:
        pred_label, prob = predict_eeg_recording(edf_path, model_name=model_name, threshold=threshold)
        pred_prob_list.append(prob)
        votes.append(int(prob >= threshold))
        print(f"Prediction from {model_name}: label={pred_label}, probability={prob}")
    
    if ensemble_method.lower() == "average":
        # Soft voting: average the probabilities.
        avg_prob = np.mean(pred_prob_list)
        final_class = int(avg_prob >= threshold)
        print("Averaged probability:", avg_prob)
        return final_class, avg_prob
    elif ensemble_method.lower() == "voting":
        # Hard voting: majority decision.
        final_class = int(round(np.mean(votes)))
        print("Votes from each model:", votes)
        return final_class, votes
    else:
        raise ValueError("Ensemble method must be either 'average' or 'voting'.")
