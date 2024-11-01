import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold
import os
import re
import datetime

# GPUが認識されているか確認
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 音声ファイルのパスをリストに追加
audio_data_dir = './traindata/'
instrument_list = ['trombone']
audio_dirs = [os.path.join(audio_data_dir, instrument) for instrument in instrument_list]
audio_files = [f for audio_dir in audio_dirs for f in os.listdir(audio_dir) if f.endswith('.mp3')]
labels = []

# 強度ラベルを抽出
def extract_intensity(file_name):
    match = re.search(r'_(mezzo-forte|pianissimo|piano|mezzo-piano|forte|fortissimo)_', file_name)
    if match:
        return match.group(1)
    else:
        print(f"ラベルをファイル名 {file_name} から抽出できませんでした。")
        return None

# ユニークなラベルを取得し、数値ラベルに変換
unique_labels = sorted({extract_intensity(f) for f in audio_files if extract_intensity(f) is not None})
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# メルスペクトログラムを計算する関数
def get_mel_spectrogram(file_path, n_mels=32, n_fft=512, hop_length=128):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# 音声データを読み込み、前処理
spectrograms = []
labels = []
max_length = 0

# メルスペクトログラムの最大長を取得
for audio_dir in audio_dirs:
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(audio_dir, file_name)
            mel_spectrogram = get_mel_spectrogram(file_path)
            max_length = max(max_length, mel_spectrogram.shape[1])

for audio_dir in audio_dirs:
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(audio_dir, file_name)
            mel_spectrogram = get_mel_spectrogram(file_path)
            if mel_spectrogram.shape[1] < max_length:
                pad_width = max_length - mel_spectrogram.shape[1]
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            intensity = extract_intensity(file_name)
            if intensity is not None:
                spectrograms.append(np.expand_dims(mel_spectrogram, axis=-1))
                labels.append(label_to_int[intensity])

spectrograms = np.array(spectrograms)
labels = np.array(labels)

# K-fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(spectrograms, labels)):
    X_train, X_val = spectrograms[train_index], spectrograms[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(spectrograms.shape[1], spectrograms.shape[2], 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # モデルの保存ディレクトリを作成
    now = datetime.datetime.now()
    save_path = f'./model/{now}/fold_{fold}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ModelCheckpointコールバックを設定
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, 'model_epoch_{epoch:03d}.keras'),
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch',
    )

    # モデルの訓練
    model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

    # モデルの概要を表示
    print(f"Fold {fold + 1}/{n_splits}")
    model.summary()