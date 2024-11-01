import numpy as np
import os
import librosa
import tensorflow as tf

# 音声ファイルからメルスペクトログラムを取得する関数
def get_mel_spectrogram(file_path, n_mels=32, n_fft=512, hop_length=128):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB
# Define label mappings
unique_labels = ['mezzo-forte', 'pianissimo', 'piano', 'cresc-decresc', 'crescendo', 'mezzo-piano', 'forte', 'fortissimo']
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_to_label = {idx: label for idx, label in enumerate(unique_labels)}

# モデルを定義する関数 (再構築用)
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# テストするための音声データのフォルダ
test_audio_dir = './traindata/trombone/'
test_audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.mp3')]

# 保存されたモデルのディレクトリを指定
model_directory = './model/20241031-120256/'  # モデルが保存されているディレクトリを指定してください
latest_model_file = max([os.path.join(model_directory, f) for f in os.listdir(model_directory) if f.endswith('.keras')],
                        key=os.path.getctime)

# モデルをロードする
model = tf.keras.models.load_model(latest_model_file)

# Correct input shape
input_shape = model.input_shape[1]

# Function to process and predict using test audio files
def predict_test_data(audio_files, test_audio_dir, input_shape):
    predictions = []
    
    for file_name in audio_files:
        file_path = os.path.join(test_audio_dir, file_name)
        mel_spectrogram = get_mel_spectrogram(file_path)
        
        # New shape after padding according to `input_shape`
        if mel_spectrogram.shape[1] < input_shape:
            pad_width = input_shape - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        
        # Flatten and ensure it matches the model input shape
        mel_spectrogram_flat = mel_spectrogram.flatten()[:input_shape].reshape(1, -1)
        
        # Predict label
        predicted_label_idx = np.argmax(model.predict(mel_spectrogram_flat), axis=1)[0]
        predicted_label = int_to_label[predicted_label_idx]
        predictions.append((file_name, predicted_label))
    
    return predictions

# Test directory and files, update path if needed
test_audio_dir = './traindata/trombone/'
test_audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.mp3')]

# Run predictions
test_predictions = predict_test_data(test_audio_files, test_audio_dir, input_shape)

# Print predictions
for file_name, predicted_label in test_predictions:
    print(f"File: {file_name}, Predicted Label: {predicted_label}")