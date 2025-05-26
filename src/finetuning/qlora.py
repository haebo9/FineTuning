# -*- coding: utf-8 -*-
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset, Features, Value
import torch
import os
import shutil
from src.config import Config
from src.functions import LossLoggerCallback  # tools.py에서 콜백 클래스 임포트

class QLoraTrainer:
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
        login(token=Config.get_hf_token())
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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            token=Config.get_hf_token()
        )

    def tokenize_dataset(self):
        self.tokenized_dataset = self.dataset.map(self.format_and_tokenize, remove_columns=self.dataset.column_names)

    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            fp16=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=True,
            hub_model_id=self.repo_id,
            hub_strategy="end"
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            callbacks=[LossLoggerCallback(save_path=self.log_path)],
            data_collator=lambda data: {
                "input_ids": torch.tensor([f["input_ids"] for f in data]),
                "attention_mask": torch.tensor([f["attention_mask"] for f in data]),
                "labels": torch.tensor([f["labels"] for f in data])
            }
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