import audioread
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# 오디오 파일을 읽고 목표 샘플 속도로 리샘플링
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
    y = scipy.signal.resample(y, int(len(y) * sr / sr_native))  # scipy 모듈 사용
    return y, sr

# 호흡 구간 감지 함수
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

# PEF 추정 함수 정의
def estimate_pef_from_amplitude(data):
    if len(data) == 0:
        return 0
    peak_amplitude = np.max(data)
    pef = peak_amplitude / 78  # 예시 상수로 조정
    return pef

# 메인 실행 함수
def main():
    audio_data = 'audio2.wav'
    x, sr = read_audio(audio_data, sr=16000)

    # 전체 신호 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(np.linspace(0, len(x) / sr, len(x)), x)
    plt.title('Waveform of Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # 호흡 구간 감지
    threshold = 10000  # 임계값 설정 (필요에 따라 조정)
    min_duration = sr * 0.2  # 최소 지속 시간 (샘플 수 기준, 0.2초)
    breathing_intervals = detect_breathing_intervals(x, sr, threshold, min_duration)

    # 감지된 호흡 구간 출력
    print("Detected breathing intervals:")
    for start, end in breathing_intervals:
        print(f"Start: {start}, End: {end}, Duration: {end - start}")

    # 각 호흡 구간의 최대 진폭 및 PEF 추정
    pefs = []
    for start, end in breathing_intervals:
        breath_segment = x[start:end]
        if len(breath_segment) == 0:
            print(f"Breath segment {start} to {end} is empty.")
            continue
        pef = estimate_pef_from_amplitude(breath_segment)
        pefs.append(pef)

    # 결과 출력 및 가장 높은 PEF 계산
    max_pef = 0
    for i, pef in enumerate(pefs):
        print(f"Breath {i + 1}: Estimated Peak Expiratory Flow (PEF) = {pef:.2f} L/min")
        if pef > max_pef:
            max_pef = pef

    print(f"Highest Estimated Peak Expiratory Flow (PEF) = {max_pef:.2f} L/min")

    # 호흡 구간 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(np.linspace(0, len(x) / sr, len(x)), x, label='Signal')
    for idx, (start, end) in enumerate(breathing_intervals):
        plt.axvline(start / sr, color='red', linestyle='--', label='Breath Start' if idx == 0 else "")
        plt.axvline(end / sr, color='blue', linestyle='--', label='Breath End' if idx == 0 else "")
    plt.title('Detected Breathing Intervals in Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 중복 레이블 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
