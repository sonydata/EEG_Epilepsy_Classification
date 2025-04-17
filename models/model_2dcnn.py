import pandas as pd
from preprocessing import preprocess
from preprocessing_2dcnn import convert_preprocessed_df_to_2d
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading metadata from Excel...")
metadata_df = pd.read_excel('eeg_metadata.xlsx') #metadata df obtained from previous extraction

print("Preprocessing data...")
processed_df = preprocess(metadata_df, num_samples=500)

print("Converting preprocessed dataframe to 2D data for CNN...")
final_df = convert_preprocessed_df_to_2d(processed_df, fs=250, nperseg=128, noverlap=64)

print("Verifying spectrogram shapes:")
print(final_df["spectrogram"].apply(lambda x: x.shape).unique())

df = final_df[["spectrogram", "epilepsy", 'subject_id']]
print("Dataframe with selected columns prepared.")

#Group-Based Train-Test Split to avoid leakage (same patient being in train and test)
print("Splitting data into train and test sets using subject_id to avoid leakage...")

def group_train_test_split(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

train_df, test_df = group_train_test_split(final_df, test_size=0.2, random_state=42)
print("Unique subjects in train:", train_df['subject_id'].unique())
print("Unique subjects in test:", test_df['subject_id'].unique())

#Prepare variables and labels for model
print("Preparing variables and labels for the model...")

X_train = np.stack(train_df['spectrogram'].values)  # shape: (n_epochs_train, 5, 65, 18)
X_test  = np.stack(test_df['spectrogram'].values)   # shape: (n_epochs_test, 5, 65, 18)

y_train = train_df['epilepsy'].values
y_test  = test_df['epilepsy'].values

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Input shape is (5, 65, 18) 
#TensorFlow expects the channel dimension last by default
print("Transposing data to channels_last format for TensorFlow...")
X_train = np.transpose(X_train, (0, 2, 3, 1))  # now shape: (n_epochs, 65, 18, 5)
X_test = np.transpose(X_test, (0, 2, 3, 1))    # now shape: (n_epochs, 65, 18, 5)
print("X_train shape:", X_train.shape)

input_shape = X_train.shape[1:] 

print("Building the model...")
model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),#add dropout
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),#add dropout
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 output classes
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Training model 
print("Starting training...")
num_epochs = 20
batch_size = 16

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test))

#Evaluate model
print("Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Accuracy: {100 * test_accuracy:.2f}%")

print("Saving metrics to Excel...")
# Extract final metrics from training history (last epoch)
metrics = {
    'Train Loss': [history.history['loss'][-1]],
    'Train Accuracy': [history.history['accuracy'][-1]],
    'Validation Loss': [history.history['val_loss'][-1]],
    'Validation Accuracy': [history.history['val_accuracy'][-1]],
    'Test Loss': [test_loss],
    'Test Accuracy': [test_accuracy]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel('model_metrics.xlsx', index=False)
print("Metrics saved to model_metrics.xlsx")