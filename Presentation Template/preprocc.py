import scipy.io
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from plot_func import plot_channel

# ----------------------------
# 1. Загрузка .mat файла
# ----------------------------
file_path = "C://Users//Пользователь//Desktop//ecog-video//Walk.mat"  # <- здесь укажи путь к своему .mat
mat = scipy.io.loadmat(file_path)
data = mat["y"]  # массив shape (164, N)

# ----------------------------
# 2. Обрезка данных
# ----------------------------
# Берем только ECoG каналы (Ch2–Ch161, индексы 1–160)
channels = data[1:161, :]



#plot_channel(channels, channel_index=1, start_idx=400, end_idx=420, title="Second Channel, first 200 points")


# ----------------------------
# 3. Предобработка
# ----------------------------
fs = 1200  # частота дискретизации

# 3.1 Notch filter на 50 Hz
f0 = 50.0
Q = 30.0
b, a = iirnotch(f0, Q, fs)
channels = filtfilt(b, a, channels, axis=1)

# 3.2 Bandpass filter 50-300 Hz
lowcut = 50.0
highcut = 300.0
b, a = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
channels = filtfilt(b, a, channels, axis=1)

# 3.3 Z-score normalization
scaler = StandardScaler()
channels = scaler.fit_transform(channels.T).T  # sklearn работает по строкам, поэтому транспонируем




#plot_channel(channels, channel_index=1, start_idx=400, end_idx=420, title="Second Channel, first 200 points")


# ----------------------------
# 4. Сохранение обработанных данных
# ----------------------------
output_npy = "processed_data.npy"
np.save(output_npy, channels)
print(f"Processed data saved as {output_npy}")

output_mat = "processed_data.mat"
scipy.io.savemat(output_mat, {"y": channels})
print(f"Processed data saved as {output_mat}")

