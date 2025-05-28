# 베이스 이미지 설정
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 모든 파일을 컨테이너의 /app 디렉토리로 복사
COPY  . /app

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 시작 시 실행할 명령어
CMD ["python3", "main.py", "--section", "haebo/Meow-Qwen2.5-7B-LoRA-fp16"]