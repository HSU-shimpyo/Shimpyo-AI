import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 오디오 파일로부터 특징을 추출
def extract_features(audio_file, sr=16000):
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        # MFCC (Mel Frequency Cepstral Coefficients) 추출
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None

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

# 학습 데이터 준비
def prepare_data(audio_folder, pef_file):
    pef_values = load_pef_values(pef_file)
    
    X = []
    y = []
    
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.m4a'):
            file_path = os.path.join(audio_folder, audio_file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(pef_values[audio_file])
    
    return np.array(X), np.array(y)

# 모델 정의
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # PEF 값을 회귀로 예측
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# 모델 학습 및 평가
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X_train.shape[1],))
    
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test MAE: {mae}')
    
    return model


def save_model(model):
    model.save('pef_model.keras')
    print("모델이 Keras V3 형식으로 'pef_model.keras'에 저장되었습니다.")


if __name__ == "__main__":
    audio_folder = './audio_files'
    pef_file = './pef_values.csv'
    
    # 학습 데이터 준비
    X, y = prepare_data(audio_folder, pef_file)
    
    # 모델 학습
    model = train_model(X, y)
    
    # 모델 저장
    save_model(model)
