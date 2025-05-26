import configparser
import argparse
import os
import logging
from datasets import load_dataset  # 데이터셋 로딩
from transformers import AutoModelForCausalLM, AutoTokenizer  # 모델 및 토크나이저 로딩
from transformers import Trainer, TrainingArguments  # 트레이너 및 학습 인자
from src.finetuning.fullft import FullFTTrainer
from src.finetuning.qlora import QLoraTrainer
from src.finetuning.lora import LoraTrainer
from src.config import Config 

def main(model_id: str, file_path: str, log_path: str, output_dir: str, repo_id: str, method: str):
    # 선택에 따른 트레이너 인스턴스 생성
    if method == 'FullFT':
        trainer = FullFTTrainer(model_id, file_path, log_path, output_dir, repo_id, dtype)
    elif method == 'QLoRA':
        trainer = QLoraTrainer(model_id, file_path, log_path, output_dir, repo_id, dtype)
    elif method == 'LoRA':
        trainer = LoraTrainer(model_id, file_path, log_path, output_dir, repo_id, dtype)
    else:
        print("Invalid method. Exiting.")
        return

    # 공통 파인튜닝 과정 실행
    trainer.initialize_environment()
    trainer.load_and_prepare_dataset()
    trainer.setup_model_and_tokenizer()
    trainer.tokenize_dataset()
    trainer_instance = trainer.setup_trainer()
    trainer.train_model(trainer_instance)
    trainer.save_and_upload_model(trainer_instance)

    print("Fine-tuning completed successfully.")

if __name__ == "__main__":
    # argparse 객체 생성
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument('--section', type=str, required=True, help='Config section to use')

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # configparser 객체 생성
    config = configparser.ConfigParser()

    # config.ini 파일 읽기
    config.read('src/config.ini')

    # 사용할 섹션 선택
    section = args.section

    # 변수 불러오기
    selected_model = config[section]['selected_model']
    selected_name = config[section]['selected_name']
    selected_method = config[section]['selected_method']
    selected_dataset = config[section]['selected_dataset']
    dtype = config[section]['dtype']

    # 기본 설정 값 (default)
    default_file_path = f"src/dataset/{selected_dataset}"
    default_log_path = f"log/Meow_{selected_name}-log.txt"
    default_output_dir = "finetuned-ktb"
    default_repo_id = f"haebo/Meow-{selected_name}_{selected_method}_{dtype}"

    print(f"repo_id : {default_repo_id}")

    # main 함수 호출
    main(selected_model, default_file_path, default_log_path, default_output_dir, default_repo_id, selected_method, dtype)

    #   python3 src/main.py --section haebo/Meow-Qwen2.5-7B-LoRA-fp16