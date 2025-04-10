import mne
import numpy as np

from typing import List, Optional
import random


def extract_random_segment(raw: mne.io.Raw, duration: float = 60.0, 
                          random_state: Optional[int] = None) -> mne.io.Raw:
    """
    Extract a random segment of specified duration from a raw MNE file.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The raw MNE object
    duration : float
        Duration of the segment to extract in seconds
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    mne.io.Raw
        A cropped raw object containing only the random segment
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Get the total duration of the raw file
    total_duration = raw.times[-1]
    
    # Ensure the raw file is long enough
    if total_duration <= duration:
        raise ValueError(f"Raw file duration ({total_duration:.2f}s) is shorter than requested segment duration ({duration:.2f}s)")
    
    # Generate a random start time
    max_start = total_duration - duration
    start_time = np.random.uniform(0, max_start)
    end_time = start_time + duration
    
    # Create a copy and crop to the random segment
    raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
    
    return raw_segment


def segment_to_epochs(raw_segment: mne.io.Raw, n_segments: int = 12) -> mne.Epochs:
    """
    Convert a raw segment into fixed-length epochs.
    
    Parameters:
    -----------
    raw_segment : mne.io.Raw
        The raw segment to convert to epochs
    n_segments : int
        Number of segments to create
    
    Returns:
    --------
    mne.Epochs
        Epoch object containing the segmented data
    """
    # Calculate duration of each epoch based on total duration and number of segments
    total_duration = raw_segment.times[-1]
    epoch_duration = total_duration / n_segments
    
    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw_segment, 
        duration=epoch_duration,
        preload=True,
        reject_by_annotation=True
    )
    
    return epochs


def process_raw_files(raw_file: mne.io.Raw, 
                      eeg_cols: List[str],
                      segment_duration: float = 60.0,
                      n_segments_per_file: int = 12,
                      samples_per_segment: int = 1250,
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Process a list of raw MNE files into a batch of epochs with specific EEG channels.
    
    Parameters:
    -----------
    raw_files : mne.io.Raw
        The raw MNE object to make preds on
    eeg_cols : List[str]
        List of EEG channel names to keep
    segment_duration : float
        Duration of random segment to extract from each file in seconds
    n_segments_per_file : int
        Number of segments to create per file
    samples_per_segment : int
        Number of time samples per segment
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray
        Array of shape (len(raw_files), n_segments_per_file, len(eeg_cols), samples_per_segment)
    """
    # Initialize the output array
    X = np.zeros((n_segments_per_file, len(eeg_cols), samples_per_segment))
    
    # Define duration of each epoch based on number of segment and total duration
    epoch_duration = segment_duration / n_segments_per_file
    

    try:
        # Set different random seed for each file if random_state is provided
        file_random_state = None if random_state is None else random_state
        
        # Pick only the specified EEG channels
        available_channels = raw_file.ch_names
        print('Num of availbable ch :', len(available_channels))
        channels_to_use = [ch for ch in available_channels if ch.replace('-REF','').replace('-LE','') in eeg_cols]
        if not channels_to_use:
            raise ValueError(f"None of the specified EEG channels found in file")
        
        if len(channels_to_use) < len(eeg_cols):
            print(f"Warning: Only {len(channels_to_use)}/{len(eeg_cols)} EEG channels found in file")
            
        # Select only the required channels
        raw_eeg = raw_file.copy().pick_channels(channels_to_use)
        
        # Resample to 250Hz
        current_sfreq = int(raw_eeg.info['sfreq'])
        if current_sfreq != 250:
            print(f"🔁 Resample : {current_sfreq} Hz → {250} Hz")
            raw_eeg.resample(250)

        # Extract random segment
        raw_segment = extract_random_segment(
            raw_eeg, 
            duration=segment_duration,
            random_state=file_random_state
        )
        
        # Convert to epochs
        epochs = segment_to_epochs(raw_segment, n_segments=n_segments_per_file)
        
        # Get the data as array
        epoch_data = epochs.get_data()
        
        # Ensure the data has the correct number of time samples
        if epoch_data.shape[2] != samples_per_segment:
            # Resample if necessary
            resampling_freq = samples_per_segment / (epoch_duration / n_segments_per_file)
            raw_segment.resample(resampling_freq)
            epochs = segment_to_epochs(raw_segment, n_segments=n_segments_per_file)
            epoch_data = epochs.get_data()
        
        # Store in the output array
        X[:, :len(channels_to_use), :] = epoch_data
        
    except Exception as e:
        print(f"Error processing file {str(e)}")
        # Keep zeros in the output array for this file
    
    return X

# Standardize the data per channel :
def standardize_data(X: np.ndarray) -> np.ndarray:
    """
    Standardize the data along the last axis (time samples).
    
    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_segments, n_channels, n_time_samples)
    
    Returns:
    --------
    np.ndarray
        Standardized data
    """
    # Compute mean and std for each channel across all segments and samples
    mean = np.mean(X, axis=(1), keepdims=True)
    std = np.std(X, axis=(1), keepdims=True)
    print(X.shape)
    
    # Standardize the data
    X_standardized = (X - mean) / std
    
    return X_standardized


# Compute Correlation Matrix
def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix for the data.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_segments, n_channels, n_time_samples)
    
    Returns:
    --------
    np.ndarray
        Correlation matrices of shape (n_samples, n_segments, n_channels, n_channels)
    """
    # Declare corr_matrix np array of shape (n_samples, n_segments,n_channels, n_channels)
    corr_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for j in range(X.shape[0]): # for each segment 5 secs
        # Compute the correlation matrix
        temp = np.corrcoef(X[j])
        corr_matrix[j] = np.nan_to_num(temp)
    
    return corr_matrix



# Discard bottom triangle from the matrix:
def discard_bottom_triangle(matrix):
    """
    Discard the bottom triangle of a square matrix.

    Parameters:
    - matrix (numpy.ndarray): The input square matrix.

    Returns:
    - numpy.ndarray: The matrix with the bottom triangle discarded.
    """
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # Apply the mask to the matrix
    upper_triangle = np.where(mask, matrix, 0)
    
    return upper_triangle

def extract_upper_triangle(corr_matrices):
    """
    Extract upper triangles from correlation matrices
    
    Args:
        corr_matrices: numpy array of shape (n_sample, n_segments, n_channels, n_channels)
        
    Returns:
        numpy array of shape (n_segments, n_features) where n_features = n_channels*(n_channels-1)/2
    """
    n_segments, n_channels, = corr_matrices.shape[0], corr_matrices.shape[1]
    n_features = n_channels * (n_channels - 1) // 2
    
    flattened = np.zeros((n_segments, n_features))
    
    for j in range(n_segments):
        # Get upper triangle indices (excluding diagonal)
        upper_indices = np.triu_indices(n_channels, k=1)
        # Extract values
        flattened[j] = corr_matrices[j][upper_indices]

    return flattened