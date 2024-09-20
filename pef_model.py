import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 오디오 파일로부터 특징을 추출
def extract_features(audio_file, sr=16000):
    y, sr = librosa.load(audio_file, sr=sr)
    
    # MFCC (Mel Frequency Cepstral Coefficients) 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# PEF 측정값을 로드 (CSV 또는 JSON)
def load_pef_values(pef_file):
    if pef_file.endswith('.csv'):
        df = pd.read_csv(pef_file)
        pef_dict = dict(zip(df['filename'], df['pef']))
    elif pef_file.endswith('.json'):
        pef_dict = pd.read_json(pef_file).to_dict()
    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")
    return pef_dict

# 데이터 증강
def augment_data(data, sr):
    # 피치 변경
    pitch_shifted = librosa.effects.pitch_shift(data, sr=sr, n_steps=2)
    
    # 시간 이동
    time_shifted = np.roll(data, sr // 10)
    
    # 소음 추가
    noise = np.random.randn(len(data)) * 0.005
    noisy_data = data + noise
    
    return [data, pitch_shifted, time_shifted, noisy_data]

# 학습 데이터 준비
def prepare_data(audio_folder, pef_file, augment=True):
    pef_values = load_pef_values(pef_file)
    
    X = []
    y = []
    
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.m4a'):
            file_path = os.path.join(audio_folder, audio_file)
            y_data, sr = librosa.load(file_path, sr=16000)
            
            if augment:
                augmented_versions = augment_data(y_data, sr)
            else:
                augmented_versions = [y_data]

            for augmented_data in augmented_versions:
                features = librosa.feature.mfcc(y=augmented_data, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(features.T, axis=0)
                X.append(mfccs_mean)
                y.append(pef_values[audio_file])
    
    return np.array(X), np.array(y)

# CNN 모델 정의
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # PEF 값을 회귀로 예측
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# 학습 및 평가
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X_train.shape[1],))
    
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test MAE: {mae}')
    
    return model

if __name__ == "__main__":
    audio_folder = './audio_files'  # 오디오 파일 폴더
    pef_file = './pef_values.csv'  # PEF 측정값 파일
    
    # 데이터 준비
    X, y = prepare_data(audio_folder, pef_file, augment=True)
    
    # 모델 학습
    model = train_model(X, y)
    
    # Keras 모델 저장
    model.save('pef_prediction_model.h5')
    print("Keras 모델이 'pef_prediction_model.h5'로 저장되었습니다.")