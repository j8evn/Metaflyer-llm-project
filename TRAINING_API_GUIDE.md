# Training API 가이드

API를 통해 LLM 학습 작업을 시작하고 관리하는 완전한 가이드입니다.

## 목차
1. [개요](#개요)
2. [서버 시작](#서버-시작)
3. [API 사용법](#api-사용법)
4. [학습 모니터링](#학습-모니터링)
5. [실전 예제](#실전-예제)

---

## 개요

Training API는 다음 기능을 제공합니다:

✅ **SFT 학습** - API로 일반 파인튜닝 시작
✅ **DPO 학습** - API로 강화학습 시작  
✅ **작업 관리** - 학습 작업 목록, 조회, 취소
✅ **로그 모니터링** - 실시간 학습 로그 확인
✅ **데이터 업로드** - API로 데이터셋 업로드

### 추론 API vs Training API

| 기능 | API 서버 | 포트 |
|------|----------|------|
| **추론** (Inference) | `api_server.py` | 8000 |
| **학습** (Training) | `training_api.py` | 8001 |

---

## 서버 시작

### 1. Training API 서버 시작

```bash
python src/training_api.py --port 8001
```

### 2. 개발 모드 (자동 리로드)

```bash
python src/training_api.py --port 8001 --reload
```

서버가 시작되면:
- **API 문서**: http://localhost:8001/docs
- **작업 목록**: http://localhost:8001/jobs

---

## API 사용법

### 1. SFT 학습 시작

**POST** `/train/sft`

```bash
curl -X POST http://localhost:8001/train/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "dataset_path": "data/train.json",
    "output_dir": "outputs/my_model",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "use_lora": true
  }'
```

**응답:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "pending",
  "message": "SFT 학습 작업이 생성되었습니다"
}
```

### 2. DPO 학습 시작

**POST** `/train/dpo`

```bash
curl -X POST http://localhost:8001/train/dpo \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "outputs/sft_model",
    "dataset_path": "data/preference_train.json",
    "output_dir": "outputs/dpo_model",
    "num_epochs": 1,
    "batch_size": 4,
    "learning_rate": 5e-7,
    "beta": 0.1,
    "use_lora": true
  }'
```

### 3. 작업 목록 조회

**GET** `/jobs`

```bash
curl http://localhost:8001/jobs
```

**응답:**
```json
[
  {
    "job_id": "a1b2c3d4",
    "training_type": "sft",
    "status": "running",
    "config": {...},
    "created_at": "2024-01-01T10:00:00",
    "started_at": "2024-01-01T10:00:05",
    "log_file": "outputs/training_logs/job_a1b2c3d4.log",
    "output_dir": "outputs/my_model"
  }
]
```

### 4. 특정 작업 조회

**GET** `/jobs/{job_id}`

```bash
curl http://localhost:8001/jobs/a1b2c3d4
```

### 5. 작업 로그 조회

**GET** `/jobs/{job_id}/logs?tail=50`

```bash
# 마지막 50줄
curl http://localhost:8001/jobs/a1b2c3d4/logs?tail=50

# 전체 로그
curl http://localhost:8001/jobs/a1b2c3d4/logs?tail=0
```

### 6. 작업 취소

**POST** `/jobs/{job_id}/cancel`

```bash
curl -X POST http://localhost:8001/jobs/a1b2c3d4/cancel
```

### 7. 데이터셋 업로드

**POST** `/upload/dataset`

```bash
curl -X POST http://localhost:8001/upload/dataset \
  -F "file=@data/my_dataset.json"
```

---

## 학습 모니터링

### Python으로 모니터링

```python
import requests
import time

def monitor_training(job_id, base_url="http://localhost:8001"):
    """학습 진행 상황 모니터링"""
    while True:
        response = requests.get(f"{base_url}/jobs/{job_id}")
        job = response.json()
        
        status = job['status']
        print(f"상태: {status}")
        
        if status in ['completed', 'failed', 'cancelled']:
            break
        
        time.sleep(5)  # 5초마다 확인
    
    # 최종 로그 출력
    response = requests.get(f"{base_url}/jobs/{job_id}/logs?tail=20")
    logs = response.json()
    
    print("\n최종 로그:")
    for log in logs['logs']:
        print(log)

# 사용
monitor_training("a1b2c3d4")
```

### 테스트 스크립트 사용

```bash
# SFT 학습 시작 및 자동 모니터링
python scripts/test_training_api.py --mode sft

# 기존 작업 모니터링
python scripts/test_training_api.py --mode monitor --job_id a1b2c3d4

# 작업 목록
python scripts/test_training_api.py --mode list
```

---

## 실전 예제

### 예제 1: 전체 파이프라인

```python
import requests
import time

BASE_URL = "http://localhost:8001"

# 1. SFT 학습 시작
print("1. SFT 학습 시작...")
response = requests.post(
    f"{BASE_URL}/train/sft",
    json={
        "model_name": "gpt2",
        "dataset_path": "data/train.json",
        "output_dir": "outputs/sft_model",
        "num_epochs": 3,
        "batch_size": 4,
        "use_lora": True
    }
)

sft_job_id = response.json()['job_id']
print(f"SFT 작업 ID: {sft_job_id}")

# 2. SFT 완료 대기
print("\n2. SFT 학습 대기...")
while True:
    response = requests.get(f"{BASE_URL}/jobs/{sft_job_id}")
    status = response.json()['status']
    print(f"   상태: {status}")
    
    if status == 'completed':
        print("   ✓ SFT 완료!")
        break
    elif status == 'failed':
        print("   ✗ SFT 실패!")
        exit(1)
    
    time.sleep(10)

# 3. DPO 학습 시작
print("\n3. DPO 학습 시작...")
response = requests.post(
    f"{BASE_URL}/train/dpo",
    json={
        "model_name": "outputs/sft_model",
        "dataset_path": "data/preference_train.json",
        "output_dir": "outputs/dpo_model",
        "num_epochs": 1,
        "batch_size": 4,
        "beta": 0.1,
        "use_lora": True
    }
)

dpo_job_id = response.json()['job_id']
print(f"DPO 작업 ID: {dpo_job_id}")

# 4. DPO 완료 대기
print("\n4. DPO 학습 대기...")
while True:
    response = requests.get(f"{BASE_URL}/jobs/{dpo_job_id}")
    status = response.json()['status']
    print(f"   상태: {status}")
    
    if status == 'completed':
        print("   ✓ DPO 완료!")
        break
    elif status == 'failed':
        print("   ✗ DPO 실패!")
        exit(1)
    
    time.sleep(10)

print("\n전체 파이프라인 완료! 🎉")
print(f"최종 모델: outputs/dpo_model")
```

### 예제 2: 배치 학습

```python
import requests

BASE_URL = "http://localhost:8001"

# 여러 설정으로 동시 학습
configs = [
    {
        "model_name": "gpt2",
        "dataset_path": "data/train.json",
        "output_dir": "outputs/model_lr_1e-5",
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "use_lora": True
    },
    {
        "model_name": "gpt2",
        "dataset_path": "data/train.json",
        "output_dir": "outputs/model_lr_2e-5",
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "use_lora": True
    },
    {
        "model_name": "gpt2",
        "dataset_path": "data/train.json",
        "output_dir": "outputs/model_lr_5e-5",
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "use_lora": True
    }
]

job_ids = []

for config in configs:
    response = requests.post(f"{BASE_URL}/train/sft", json=config)
    job_id = response.json()['job_id']
    job_ids.append(job_id)
    print(f"작업 시작: {job_id} (lr={config['learning_rate']})")

print(f"\n총 {len(job_ids)}개의 작업이 실행 중입니다.")
```

### 예제 3: 웹 대시보드

```html
<!DOCTYPE html>
<html>
<head>
    <title>LLM Training Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .job { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
        .running { background-color: #ffffcc; }
        .completed { background-color: #ccffcc; }
        .failed { background-color: #ffcccc; }
    </style>
</head>
<body>
    <h1>LLM Training Dashboard</h1>
    <div id="jobs"></div>

    <script>
        async function loadJobs() {
            const response = await fetch('http://localhost:8001/jobs');
            const jobs = await response.json();
            
            const container = document.getElementById('jobs');
            container.innerHTML = '';
            
            jobs.forEach(job => {
                const div = document.createElement('div');
                div.className = `job ${job.status}`;
                div.innerHTML = `
                    <h3>작업 ${job.job_id}</h3>
                    <p>타입: ${job.training_type}</p>
                    <p>상태: ${job.status}</p>
                    <p>생성: ${job.created_at}</p>
                    ${job.output_dir ? `<p>출력: ${job.output_dir}</p>` : ''}
                `;
                container.appendChild(div);
            });
        }

        // 5초마다 업데이트
        setInterval(loadJobs, 5000);
        loadJobs();
    </script>
</body>
</html>
```

---

## 작업 상태

| 상태 | 설명 |
|------|------|
| `pending` | 작업이 생성되었으나 아직 시작 안 됨 |
| `running` | 학습 진행 중 |
| `completed` | 학습 완료 |
| `failed` | 학습 실패 |
| `cancelled` | 사용자가 취소 |

---

## 주의사항

### 1. 동시 학습

여러 학습 작업을 동시에 실행할 수 있지만, GPU 메모리를 고려하세요:

```python
# GPU 메모리가 충분하지 않으면 순차 실행
job1 = start_training(config1)
wait_for_completion(job1)

job2 = start_training(config2)
wait_for_completion(job2)
```

### 2. 로그 파일

모든 학습 로그는 `outputs/training_logs/` 에 저장됩니다:
```
outputs/training_logs/
├── job_a1b2c3d4.log
├── job_e5f6g7h8.log
└── ...
```

### 3. 서버 재시작

서버를 재시작하면 진행 중인 작업 정보가 손실됩니다. 프로덕션 환경에서는 데이터베이스를 사용하세요.

---

## 두 API 서버 함께 실행

### 터미널 1: Inference API (추론)

```bash
python src/api_server.py \
    --model_path "outputs/checkpoints/final_model" \
    --port 8000
```

### 터미널 2: Training API (학습)

```bash
python src/training_api.py --port 8001
```

### 전체 워크플로우

```bash
# 1. Training API로 학습
curl -X POST http://localhost:8001/train/sft \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", ...}'

# 2. 학습 완료 대기

# 3. Inference API로 모델 로딩
curl -X POST "http://localhost:8000/load_model?model_path=outputs/my_model"

# 4. 추론 실행
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"instruction": "테스트 질문"}'
```

---

## 트러블슈팅

### 문제 1: 작업이 pending에서 멈춤

- 로그 확인: `curl http://localhost:8001/jobs/{job_id}/logs`
- 서버 로그 확인: 터미널 출력 확인

### 문제 2: 작업이 실패함

```bash
# 상세 정보 확인
curl http://localhost:8001/jobs/{job_id}

# 로그 확인
curl http://localhost:8001/jobs/{job_id}/logs?tail=0
```

### 문제 3: GPU 메모리 부족

학습 설정에서 배치 크기 감소:
```json
{
  "batch_size": 1,
  "use_lora": true
}
```

---

더 자세한 내용은 http://localhost:8001/docs 의 API 문서를 참조하세요!

