# Python 3.9 slim 이미지를 기반으로 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일들 복사
COPY requirements.txt requirements.txt
COPY my_flask.py my_flask.py
COPY pef_model.py pef_model.py
COPY pef_prediction_model.h5 pef_prediction_model.h5
COPY pef_values.csv pef_values.csv

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 실행
CMD ["python", "my_flask.py"]
