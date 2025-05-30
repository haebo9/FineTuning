# config.py
from dotenv import load_dotenv
import os
import torch

# .env 파일 로드
load_dotenv()

# 환경 변수 관리
class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # config instance 정의 없이도 호출하기 위해 @staticmethod 데코레이터 사용
    @staticmethod 
    def get_hf_token():
        if Config.HF_TOKEN is None:
            raise ValueError("HF_TOKEN is not set in the environment variables.")
        return Config.HF_TOKEN

    @staticmethod
    def get_device():
        return Config.DEVICE