# Python 3.9 slim 이미지를 기반으로 설정
FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && pip install h5py \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 코드 복사
COPY . .

# 불필요한 파일 삭제 (최종 이미지 크기 최적화)
RUN apt-get remove -y gcc g++ pkg-config && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Flask 애플리케이션 실행
CMD ["python", "my_flask.py"]
