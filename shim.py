import audioread
import numpy as np
import scipy.signal
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

def download_audio_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def read_audio(file_path, sr=16000):
    y = []
    with audioread.audio_open(file_path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        for frame in input_file:
            samples = np.frombuffer(frame, dtype=np.int16)
            samples = samples.reshape(-1, n_channels)
            samples = samples.mean(axis=1)
            y.extend(samples)
    y = np.array(y, dtype=np.float32)
    y = scipy.signal.resample(y, int(len(y) * sr / sr_native))
    return y, sr

def detect_breathing_intervals(data, sr, threshold, min_duration):
    intervals = []
    start = None
    for i, sample in enumerate(data):
        if sample > threshold and start is None:
            start = i
        elif sample < threshold and start is not None:
            if i - start > min_duration:
                intervals.append((start, i))
                start = None
    return intervals

def estimate_pef_from_amplitude(data):
    if len(data) == 0:
        return 0
    peak_amplitude = np.max(data)
    pef = peak_amplitude / 78  # 예시 상수로 조정
    return pef

def process_audio_file(audio_file_url, local_filename):
    download_audio_file(audio_file_url, local_filename)
    x, sr = read_audio(local_filename, sr=16000)

    threshold = 10000
    min_duration = sr * 0.2
    breathing_intervals = detect_breathing_intervals(x, sr, threshold, min_duration)

    pefs = []
    for start, end in breathing_intervals:
        breath_segment = x[start:end]
        pef = estimate_pef_from_amplitude(breath_segment)
        pefs.append(pef)

    max_pef = max(pefs) if pefs else 0
    return max_pef

@app.route('/analyze', methods=['POST'])
def analyze_files():
    # Spring Boot 서버로부터 S3 URL을 JSON 형태로 받음
    file_urls = request.json
    
    pef_values = {}

    # 각 파일에 대해 PEF 계산
    for i, (key, url) in enumerate(file_urls.items()):
        local_filename = f'audio_{i+1}.wav'
        pef = process_audio_file(url, local_filename)
        pef_values[f'pef_{i+1}'] = pef

    # 계산된 PEF 값을 JSON으로 반환
    return jsonify(pef_values), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # 로컬에서 Flask 서버를 5000번 포트로 실행