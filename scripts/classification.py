import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models
import os
import matplotlib.pyplot as plt

def get_mel_spectrogram(file_path, n_mels=32, n_fft=512, hop_length=128):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    mel_spectrogram = S_dB
    return mel_spectrogram

model_dir = './model'
loaded_model = tf.keras.models.load_model(os.path.join(model_dir, '2024-11-01 11:08:15.977496/fold_0/model_epoch_020.keras'))

def classify_audio(file_path, model):
    mel_spectrogram = get_mel_spectrogram(file_path)
    
    # モデルの入力形状を取得
    target_shape = (model.input_shape[1], model.input_shape[2])

    # パディングまたはトリミングを行う
    current_length = mel_spectrogram.shape[1]
    target_length = target_shape[1]
    
    if current_length < target_length:
        pad_width = target_length - current_length
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :target_length]

    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # バッチ次元を追加

    # モデルの入力形状にリサイズ
    mel_spectrogram = tf.image.resize(mel_spectrogram, (target_shape[0], target_shape[1]))
    
    prediction = model.predict(mel_spectrogram)
    predicted_label = unique_labels[np.argmax(prediction)]
   
    return predicted_label

test_file_path = './traindata/trombone/trombone_B3_1_pianissimo_normal.mp3'

unique_labels = [
    'mezzo-forte',
    'pianissimo',
    'piano',
    'mezzo-piano',
    'forte',
    'fortissimo',
]  # ラベルのリストを設定

predicted_label = classify_audio(test_file_path, loaded_model)

print(f"Predicted label: {predicted_label}")