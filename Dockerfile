# Python 3.9 slim 이미지를 기반으로 설정
FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 목록만 먼저 복사
COPY requirements.txt .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 코드 복사
COPY . .

# Flask 애플리케이션 실행
CMD ["python", "my_flask.py"]
