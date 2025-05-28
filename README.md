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
## 라이브러리 버전
| 라이브러리 | 버전 | 목적 |
|------------|------|------|
| `torch` | 2.6.0 | PyTorch. 딥러닝 모델 학습 및 GPU 연산 지원 |
| `transformers` | 4.51.3 | LLM 모델 로딩 및 추론용 (GPT, Qwen 등) |
| `peft` | 0.15.2 | LoRA, QLoRA 등 경량화 파인튜닝 지원 |
| `datasets` | 3.6.0 | 다양한 형식의 학습 데이터셋 로딩 및 전처리 |
| `numpy` | 2.0.2 | 수치 계산 및 배열 연산의 기본 라이브러리 |
| `python-dotenv` | 1.1.0 | `.env` 파일로부터 환경 변수 로딩 지원 |
| `huggingface_hub[hf_xet]` | 최신 | HF Hub 모델 다운로드 및 업로드<br>`hf_xet`은 Xformers 최적화 포함 |
| `accelerate` | >=0.27.2 | 멀티-GPU/TPU 학습을 위한 추상화 도구 |
| `sentencepiece` | 최신 | subword 기반 tokenizer (T5, Qwen 등에서 필수) |


## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/haebo9/FineTuning.git
```

2. 가상환경 생성 및 활성화
   - **Conda**:
     ```bash
     conda create --name finetuning python=3.12
     conda activate finetuning
     ```
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
   - **Conda 환경에서 pip 사용**:
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


## 도커 사용 방법

이 프로젝트는 도커를 사용하여 쉽게 실행할 수 있습니다. 아래의 단계를 따라 도커 이미지를 빌드하고 컨테이너를 실행하세요.

### 도커 이미지 빌드

1. 도커가 설치되어 있는지 확인합니다.
2. 프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 도커 이미지를 빌드합니다:
   ```bash
   docker build -t my-python-app .
   ```

### 도커 컨테이너 실행

1. 빌드된 이미지를 기반으로 컨테이너를 실행합니다:
   ```bash
   docker run -d my-python-app
   ```

2. 컨테이너가 실행되면, `main.py` 스크립트가 자동으로 실행됩니다:
   ```bash
   python3 main.py --section haebo/Meow-Qwen2.5-7B-LoRA-fp16
   ```

3. 실행 중인 컨테이너의 로그를 확인하여 스크립트가 정상적으로 작동하는지 확인할 수 있습니다:
   ```bash
   docker logs <container_id>
   ```