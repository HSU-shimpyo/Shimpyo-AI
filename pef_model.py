import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 오디오 데이터 증강 함수
def add_white_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def time_shift(data, shift_max=0.2):
    shift = np.random.randint(int(len(data) * shift_max))
    direction = np.random.choice(['left', 'right'])
    if direction == 'right':
        shift = -shift
    augmented_data = np.roll(data, shift)
    return augmented_data

def my_pitch_shift(data, sr, n_steps):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate):
    # 속도 조절
    return librosa.effects.time_stretch(data, rate=rate)

# 오디오 파일로부터 특징을 추출
def extract_features(y, sr, n_mfcc=13):
    try:
        # MFCC (Mel Frequency Cepstral Coefficients) 추출
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features: {e}")
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
def prepare_data(audio_folder, pef_file, augment=True):
    pef_values = load_pef_values(pef_file)
    
    X = []
    y_list = []
    
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.m4a'):
            file_path = os.path.join(audio_folder, audio_file)
            try:
                y_data, sr = librosa.load(file_path, sr=16000)
                # 원본 데이터의 특징 추출
                features = extract_features(y_data, sr)
                if features is not None:
                    X.append(features)
                    y_list.append(pef_values[audio_file])
                
                if augment:
                    # 데이터 증강 적용
                    augmented_data_list = []

                    # 1. 화이트 노이즈 추가
                    y_noise = add_white_noise(y_data)
                    augmented_data_list.append(y_noise)

                    # 2. 시간 이동
                    y_shift = time_shift(y_data)
                    augmented_data_list.append(y_shift)

                    # 3. 피치 변환 (반음 올리기)
                    y_pitch_up = my_pitch_shift(y_data, sr=sr, n_steps=2)
                    augmented_data_list.append(y_pitch_up)

                    # 4. 피치 변환 (반음 내리기)
                    y_pitch_down = my_pitch_shift(y_data, sr=sr, n_steps=-2)
                    augmented_data_list.append(y_pitch_down)

                    # 5. 속도 조절 (빠르게)
                    y_speed_up = time_stretch(y_data, rate=1.25)
                    augmented_data_list.append(y_speed_up)

                    # 6. 속도 조절 (느리게)
                    y_slow_down = time_stretch(y_data, rate=0.8)
                    augmented_data_list.append(y_slow_down)

                    # 각 증강 데이터에 대해 특징 추출
                    for aug_data in augmented_data_list:
                        features_aug = extract_features(aug_data, sr)
                        if features_aug is not None:
                            X.append(features_aug)
                            y_list.append(pef_values[audio_file])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return np.array(X), np.array(y_list)

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
    X, y = prepare_data(audio_folder, pef_file, augment=True)

    # 모델 학습
    model = train_model(X, y)

    # 모델 저장
    save_model(model)
