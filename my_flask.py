import os
from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf  # TensorFlow 사용
import logging

# 로그 설정
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)

# 모델 파일 경로 설정
model_path = os.path.join(os.path.dirname(__file__), 'pef_prediction_model.h5')

# TensorFlow 모델 로드
model = tf.keras.models.load_model(model_path)  # 전체 모델 로드

# 오디오 파일로부터 특징을 추출
def extract_features(audio_file, sr=16000):
    y, sr = librosa.load(audio_file, sr=sr)
    
    # MFCC (Mel Frequency Cepstral Coefficients) 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# TensorFlow 모델을 사용하여 예측 수행
def predict_pef(features):
    features = np.array(features, dtype=np.float32)
    features = np.expand_dims(features, axis=0)

    # 모델을 사용하여 PEF 예측
    predicted_pef = model.predict(features)
    return float(predicted_pef[0][0])

# 여러 파일 업로드 처리
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')
    
    if not files:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400
    
    pef_results = {}  # PEF 결과를 저장할 딕셔너리
    
    for idx, file in enumerate(files, start=1):
        if file and file.filename.endswith('.m4a'):
            file_path = os.path.join('./uploads', file.filename)
            file.save(file_path)
            
            features = extract_features(file_path)
            
            # TensorFlow 모델을 사용하여 PEF 예측
            predicted_pef_value = predict_pef(features)
            
            # 로그 기록: 예측 결과 기록
            logging.info(f"Predicted PEF value: {predicted_pef_value} for file: {file.filename}")
            
            # 결과 저장
            pef_results[f"pef_{idx}"] = predicted_pef_value
        else:
            return jsonify({"error": f"{file.filename}는 m4a 파일이 아닙니다."}), 400

    return jsonify(pef_results), 200

if __name__ == '__main__':
    os.makedirs('./uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
