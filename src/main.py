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
        trainer = FullFTTrainer(model_id, file_path, log_path, output_dir, repo_id)
    elif method == 'QLoRA':
        trainer = QLoraTrainer(model_id, file_path, log_path, output_dir, repo_id)
    elif method == 'LoRA':
        trainer = LoraTrainer(model_id, file_path, log_path, output_dir, repo_id)
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

# if __name__ == "__main__":
#     # 선택값 (can modify)
#     selected_model = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B" # 베이스 모델
#     selected_name = "HyperCLOVAX-1.5B" # 모델 별칭 
#     selected_method = "FullFT"  # 파인튜닝 기법 선택
#     selected_dataset = "posts-0515.jsonl" # 사용할 데이터 타입
#     dtype = ""

#     # 기본 설정 값 (defualt)
#     default_file_path = f"./dataset/{selected_dataset}"
#     default_log_path = f"./log/Meow_{selected_name}-log.txt"
#     default_output_dir = "./finetuned-ktb"
#     default_repo_id = f"haebo/Meow-{selected_name}_{selected_method}_{dtype}"
    

#     # main 함수 호출
#     main(selected_model, default_file_path, default_log_path, default_output_dir, default_repo_id, selected_method)

if __name__ == "__main__":
    # 허깅페이스 토큰 정의
    hf_token = Config.get_hf_token()

    # 선택값 (can modify)
    selected_model = "Qwen/Qwen2.5-7B-Instruct"  # 베이스 모델
    selected_name = "Qwen2.5-7B"  # 모델 별칭
    selected_method = "LoRA"  # 파인튜닝 기법 선택
    selected_dataset = "posts-0515.jsonl"  # 사용할 데이터 타입
    dtype = "fp16"  # 데이터 타입에 대한 추가 정보가 필요할 경우 사용

    # 기본 설정 값 (default)
    default_file_path = f"src/dataset/{selected_dataset}"
    default_log_path = f"log/Meow_{selected_name}-log.txt"
    default_output_dir = "finetuned-ktb"
    default_repo_id = f"haebo/Meow-{selected_name}_{selected_method}_{dtype}"

    # main 함수 호출
    main(selected_model, default_file_path, default_log_path, default_output_dir, default_repo_id, selected_method)