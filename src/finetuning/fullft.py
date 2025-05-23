# -*- coding: utf-8 -*-
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Features, Value
import torch
import os
import shutil
import json
from src.config import get_hf_token
from src.functions import LossLoggerCallback  # tools.py에서 콜백 클래스 임포트

class FullFTTrainer:
    def __init__(self, model_id, file_path, log_path, output_dir, repo_id):
        self.model_id = model_id
        self.file_path = file_path
        self.log_path = log_path
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        self.api = HfApi()

    def initialize_environment(self):
        login(token=get_hf_token())
        print("✅ 현재 사용자:", self.api.whoami()["name"])

    def load_and_prepare_dataset(self):
        temp_cache_dir = "./tmp_cache"
        os.makedirs(temp_cache_dir, exist_ok=True)
        features = Features({
            "content": Value("string"),
            "emotion": Value("string"),
            "post_type": Value("string"),
            "transformed_content": Value("string")
        })
        self.dataset = load_dataset(
            "json",
            data_files=self.file_path,
            features=features,
            split="train",
            cache_dir=temp_cache_dir,
            verification_mode="no_checks"
        )
        shutil.rmtree(temp_cache_dir, ignore_errors=True)

    def format_and_tokenize(self, example):
        text = (
            f"### content:\n{example['content']}\n"
            f"### emotion:\n{example['emotion']}\n"
            f"### post_type:\n{example['post_type']}\n\n"
            f"### transformed_content:\n{example['transformed_content']}"
        )
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def tokenize_dataset(self):
        self.tokenized_dataset = self.dataset.map(self.format_and_tokenize, remove_columns=self.dataset.column_names)

    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            fp16=False,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=True,
            hub_model_id=self.repo_id,
            hub_strategy="end"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            callbacks=[LossLoggerCallback(save_path=self.log_path)],
            data_collator=data_collator
        )
        return trainer

    def train_model(self, trainer):
        trainer.train()

    def save_and_upload_model(self, trainer):
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True, private=False)
        self.api.upload_folder(folder_path=self.output_dir, repo_id=self.repo_id, repo_type="model")
        print(f"✅ 모델 업로드 완료: https://huggingface.co/{self.repo_id}")

# Main execution
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
file_path = "./dataset/posts-0515.jsonl"
log_path = "./log/Meow_HyperCLOVAX-1.5B-FullFT-fp32_log.txt"
output_dir = "./finetuned-ktb"
repo_id = "haebo/Meow-HyperCLOVAX-1.5B-FullFT-fp32"

fullft_trainer = FullFTTrainer(model_id, file_path, log_path, output_dir, repo_id)
fullft_trainer.initialize_environment()
fullft_trainer.load_and_prepare_dataset()
fullft_trainer.setup_model_and_tokenizer()
fullft_trainer.tokenize_dataset()
trainer = fullft_trainer.setup_trainer()
fullft_trainer.train_model(trainer)
fullft_trainer.save_and_upload_model(trainer)