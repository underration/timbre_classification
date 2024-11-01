# メルスペクトルを可視化するスクリプト
import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
import os
import re
import datetime
import matplotlib.pyplot as plt
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
instrument_list = [ 'french horn']
audio_dirs = [os.path.join(audio_data_dir, instrument) for instrument in instrument_list]
audio_files = [f for audio_dir in audio_dirs for f in os.listdir(audio_dir) if f.endswith('.mp3')]

labels = []

# 強度ラベルを抽出
def extract_intensity(file_name):
	match = re.search(r'_(mezzo-forte|pianissimo|piano|cresc-decresc|crescendo|mezzo-piano|forte|fortissimo|phrase_forte_glissando|phrase_pianissimo_normal|very-long_cresc-decresc)_', file_name)
	if match:
		return match.group(1)
	else:
		raise ValueError(f"ラベルをファイル名 {file_name} から抽出できませんでした。")

# ユニークなラベルを取得し、数値ラベルに変換
unique_labels = sorted({extract_intensity(f) for f in audio_files})
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
			spectrograms.append(np.expand_dims(mel_spectrogram, axis=-1))
			labels.append(label_to_int[extract_intensity(file_name)])
	spectrograms.append(np.expand_dims(mel_spectrogram, axis=-1))
	labels.append(label_to_int[extract_intensity(file_name)])

spectrograms = np.array(spectrograms)
labels = np.array(labels)

# メルスペクトログラムを可視化
def plot_spectrograms(spectrograms, labels, unique_labels, num_to_plot=5):
	plt.figure(figsize=(10, 10))
	for i in range(num_to_plot):
		plt.subplot(num_to_plot, 1, i + 1)
		spectrogram = spectrograms[i].squeeze()
		plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, spectrogram.shape[1], 0, spectrogram.shape[0]])
		plt.title(unique_labels[labels[i]])
		cbar = plt.colorbar(format='%+2.0f dB')
		cbar.mappable.set_clim(vmin=np.min(spectrogram), vmax=np.max(spectrogram))  # カラーバーの範囲を設定
		plt.xlim([0, 1100])
	plt.tight_layout()
	plt.show()
	print(spectrogram)

# メルスペクトログラムを可視化
plot_spectrograms(spectrograms, labels, unique_labels, num_to_plot=5)
