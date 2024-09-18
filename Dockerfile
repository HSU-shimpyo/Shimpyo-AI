# 베이스 이미지를 설정 (파이썬 3.9 사용)
FROM python:3.9-slim

# FFmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg

# 작업 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 파일을 컨테이너로 복사
COPY . .

# 필요한 라이브러리 설치
RUN pip install -r requirements.txt

# 서버 실행
CMD ["python3", "shim.py"]
