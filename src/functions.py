# functions.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from src.config import Config

# ------------------------------------------------------
# ✅ step 단위 loss 로그를 저장하는 콜백 클래스 (클래스명 유지)
# ------------------------------------------------------
class LossLoggerCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            with open(self.save_path, "a") as f:
                f.write(f"Step {step}: loss = {loss:.4f}\n")

# ------------------------------------------------------
# ✅ 모델 로더 클래스
# ------------------------------------------------------
class ModelLoader:
    def __init__(self):
        self.HF_TOKEN = Config.get_hf_token()
        self.DEVICE = Config.get_device()
        self.model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        print(f'device : {self.DEVICE}')
        print(f'HF_TOKEN : {True if self.HF_TOKEN else False}')
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f'model : {self.model_id}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.HF_TOKEN
        )

# ------------------------------------------------------
# ✅ 응답 생성 클래스
# ------------------------------------------------------
class ResponseGenerator:
    def __init__(self, model_id: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        input_text = (
            f"### core guidelines.\n"
            f"1. Transform speech without losing the meaning of the original text.\n"
            f"2. Use only one emoticon per sentence.\n"
            f"3. Make a clear distinction between cat and dog responses.\n"
            f"### content:\n{prompt}\n"
            f"### transformed_content:"
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        start_index = response.find("### 고양이:")
        if start_index != -1:
            response = response[start_index + len("### 고양이:"):].strip()

        return response