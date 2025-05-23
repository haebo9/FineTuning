# FineTuning
for sLLM FineTuning

```Markdown
> 테스트 모델 :
    - naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B

> 방식
    - LoRA

> 허깅페이스 아이디
    - haebo

```

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/haebo9/FineTuning.git
```

2. 가상환경 생성 및 활성화
   - **Linux/Mac**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 프로젝트 목적
이 프로젝트는 sLLM FineTuning을 위한 것으로, LoRA 방식을 사용하여 모델을 미세 조정합니다.

## 프로젝트 디렉토리 구조
```plaintext
FineTuning/
│
├── .gitignore
├── README.md
├── requirement.txt
├── src/
│   ├── config.py
│   ├── functions.py
│   ├── main.py
│   ├── dataset/
│   │   └── posts-0515.jsonl
│   └── finetuning/
│       ├── fullft.py
│       ├── qlora.py
│       └── lora.py
└── log/
```

## 주요 파일 설명

- `src/config.py`: 환경 변수와 설정을 관리하는 파일입니다.
- `src/functions.py`: 모델 로딩 및 응답 생성과 관련된 기능을 포함합니다.
- `src/main.py`: 메인 실행 파일입니다.
- `src/dataset/posts-0515.jsonl`: 데이터셋 파일로, JSONL 형식으로 저장되어 있습니다.
- `src/finetuning/fullft.py`: Full Fine-Tuning을 위한 스크립트입니다.
- `src/finetuning/qlora.py`: QLoRA를 사용한 미세 조정 스크립트입니다.
- `src/finetuning/lora.py`: LoRA를 사용한 미세 조정 스크립트입니다.

## 실행 방법

1. 환경 설정
   - `.env` 파일을 생성하고 `HF_TOKEN`을 설정합니다.

2. 모델 학습
   - `src/finetuning/fullft.py`, `src/finetuning/qlora.py`, `src/finetuning/lora.py` 중 하나를 실행하여 모델을 학습시킵니다.

3. 결과 확인
   - 학습된 모델은 Hugging Face Hub에 업로드됩니다.