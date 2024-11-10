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
instrument_list = [ 'french horn', 'trumpet']
audio_dirs = [os.path.join(audio_data_dir, instrument) for instrument in instrument_list]
audio_files = [f for audio_dir in audio_dirs for f in os.listdir(audio_dir) if f.endswith('.mp3')]
print(f'audio_files: {audio_files}')
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

os.makedirs('spectrograms', exist_ok=True)
# 各ファイルに対して処理を行う
for file in audio_files:
	try:
		instrument_match = re.search(r'(trombone|french horn|trumpet)', file)
		if instrument_match:
			instrument = instrument_match.group(1)
			figure_dir = os.path.join(audio_data_dir, instrument)
			os.makedirs(f'./spectrograms/{instrument}', exist_ok=True)
			label = extract_intensity(file)
			labels.append(label_to_int[label])
			spectrogram = get_mel_spectrogram(os.path.join(figure_dir, file))
			if spectrogram is not None:
				spectrograms.append(spectrogram)
				max_length = spectrogram.shape[1]
				# save spectrogram
				plt.figure(figsize=(10, 4))
				plt.imshow(spectrogram, origin='lower', aspect='auto')
				plt.colorbar(format='%+2.0f dB')
				plt.title(label)
				plt.tight_layout()
				plt.savefig(f'./spectrograms/{instrument}/{file}.png')
				plt.close()
		else:
			print(f"楽器名をファイル名 {file} から抽出できませんでした。")
	except ValueError as e:
		print(e)