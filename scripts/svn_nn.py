import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.metrics.pairwise import rbf_kernel
import os
import re
import datetime

# Ensure GPUs are recognized
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

configure_gpu_memory_growth()

# Set audio data directory and initialize lists for files and labels
audio_data_dir = './traindata/'
instrument_list = ['trombone']
audio_dirs = [os.path.join(audio_data_dir, instrument) for instrument in instrument_list]
audio_files = [f for audio_dir in audio_dirs for f in os.listdir(audio_dir) if f.endswith('.mp3')]

# Function to extract intensity label from file name
def extract_intensity(file_name):
    pattern = r'_(mezzo-forte|pianissimo|piano|cresc-decresc|crescendo|mezzo-piano|forte|fortissimo|phrase_forte_glissando|phrase_pianissimo_normal|very-long_cresc-decresc)_'
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Cannot extract label from file name: {file_name}")

# Create a mapping from labels to integers
unique_labels = sorted({extract_intensity(f) for f in audio_files})
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# Function to compute Mel spectrogram
def get_mel_spectrogram(file_path, n_mels=32, n_fft=512, hop_length=128):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def prepare_data(audio_dirs):
    spectrograms = []
    labels = []
    max_length = 0

    # Determine the maximum length of Mel spectrograms
    for audio_dir in audio_dirs:
        for file_name in os.listdir(audio_dir):
            if file_name.endswith('.mp3'):
                file_path = os.path.join(audio_dir, file_name)
                mel_spectrogram = get_mel_spectrogram(file_path)
                max_length = max(max_length, mel_spectrogram.shape[1])

    # Normalize and pad spectrograms
    for audio_dir in audio_dirs:
        for file_name in os.listdir(audio_dir):
            if file_name.endswith('.mp3'):
                file_path = os.path.join(audio_dir, file_name)
                mel_spectrogram = get_mel_spectrogram(file_path)
                if mel_spectrogram.shape[1] < max_length:
                    pad_width = max_length - mel_spectrogram.shape[1]
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                spectrograms.append(mel_spectrogram.flatten())
                labels.append(label_to_int[extract_intensity(file_name)])
    
    return np.array(spectrograms), np.array(labels), max_length

# Preprocess the audio data
spectrograms, labels, max_length = prepare_data(audio_dirs)

# Function for RBF kernel feature transformation
def rbf_feature_transform(X, gamma=0.1):
    K = rbf_kernel(X, X, gamma=gamma)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals[eigvals < 0] = 0
    L = eigvecs @ np.diag(np.sqrt(eigvals))
    return L

# Transform the features to RBF kernel space
L = rbf_feature_transform(spectrograms)

# Define the neural network model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create model directory and compile the model
def create_model_directory():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f'./model/{now}'
    os.makedirs(save_path, exist_ok=True)
    return save_path

model = build_model(input_shape=L.shape[1], num_classes=len(unique_labels))

# Set up model checkpointing
save_path = create_model_directory()
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join(save_path, 'model_epoch_{epoch:03d}.keras'),
    save_weights_only=False,
    save_best_only=False,
)

# Train the model
model.fit(L, labels, epochs=50, batch_size=16, validation_split=0.2, callbacks=[checkpoint_callback])

# Display the model summary
model.summary()
print("Max spectrogram length:", max_length)