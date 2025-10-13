"""
Training API 테스트 스크립트
"""

import requests
import json
import time


def test_sft_training(base_url: str = "http://localhost:8001"):
    """SFT 학습 테스트"""
    print("=" * 60)
    print("SFT 학습 작업 생성")
    print("=" * 60)
    
    data = {
        "model_name": "gpt2",
        "dataset_path": "data/train.json",
        "output_dir": "outputs/test_sft",
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 2e-5,
        "use_lora": True
    }
    
    print(f"요청 데이터:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()
    
    response = requests.post(f"{base_url}/train/sft", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 작업 생성 성공")
        print(f"작업 ID: {result['job_id']}")
        print(f"상태: {result['status']}")
        print(f"메시지: {result['message']}")
        return result['job_id']
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)
        return None


def test_dpo_training(base_url: str = "http://localhost:8001"):
    """DPO 학습 테스트"""
    print("\n" + "=" * 60)
    print("DPO 학습 작업 생성")
    print("=" * 60)
    
    data = {
        "model_name": "outputs/sft_model",
        "dataset_path": "data/preference_train.json",
        "output_dir": "outputs/test_dpo",
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 5e-7,
        "beta": 0.1,
        "use_lora": True
    }
    
    print(f"요청 데이터:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()
    
    response = requests.post(f"{base_url}/train/dpo", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 작업 생성 성공")
        print(f"작업 ID: {result['job_id']}")
        print(f"상태: {result['status']}")
        print(f"메시지: {result['message']}")
        return result['job_id']
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)
        return None


def test_list_jobs(base_url: str = "http://localhost:8001"):
    """작업 목록 조회"""
    print("\n" + "=" * 60)
    print("작업 목록 조회")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/jobs")
    
    if response.status_code == 200:
        jobs = response.json()
        print(f"총 {len(jobs)}개의 작업")
        print()
        
        for job in jobs:
            print(f"작업 ID: {job['job_id']}")
            print(f"  타입: {job['training_type']}")
            print(f"  상태: {job['status']}")
            print(f"  생성: {job['created_at']}")
            if job.get('output_dir'):
                print(f"  출력: {job['output_dir']}")
            print()
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)


def test_get_job(job_id: str, base_url: str = "http://localhost:8001"):
    """특정 작업 조회"""
    print("\n" + "=" * 60)
    print(f"작업 조회: {job_id}")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/jobs/{job_id}")
    
    if response.status_code == 200:
        job = response.json()
        print(f"작업 ID: {job['job_id']}")
        print(f"타입: {job['training_type']}")
        print(f"상태: {job['status']}")
        print(f"생성 시간: {job['created_at']}")
        
        if job.get('started_at'):
            print(f"시작 시간: {job['started_at']}")
        if job.get('completed_at'):
            print(f"완료 시간: {job['completed_at']}")
        if job.get('error_message'):
            print(f"에러: {job['error_message']}")
        
        print(f"\n설정:")
        print(json.dumps(job['config'], indent=2, ensure_ascii=False))
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)


def test_get_logs(job_id: str, base_url: str = "http://localhost:8001", tail: int = 20):
    """작업 로그 조회"""
    print("\n" + "=" * 60)
    print(f"작업 로그: {job_id} (마지막 {tail}줄)")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/jobs/{job_id}/logs?tail={tail}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"총 {result['total_lines']}줄")
        print("\n로그:")
        print("-" * 60)
        for log in result['logs']:
            print(log)
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)


def test_cancel_job(job_id: str, base_url: str = "http://localhost:8001"):
    """작업 취소"""
    print("\n" + "=" * 60)
    print(f"작업 취소: {job_id}")
    print("=" * 60)
    
    response = requests.post(f"{base_url}/jobs/{job_id}/cancel")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
    else:
        print(f"✗ 오류: {response.status_code}")
        print(response.text)


def monitor_job(job_id: str, base_url: str = "http://localhost:8001", interval: int = 5):
    """작업 모니터링"""
    print("\n" + "=" * 60)
    print(f"작업 모니터링: {job_id}")
    print("=" * 60)
    print("Ctrl+C로 중지")
    print()
    
    try:
        while True:
            response = requests.get(f"{base_url}/jobs/{job_id}")
            
            if response.status_code == 200:
                job = response.json()
                status = job['status']
                
                print(f"[{time.strftime('%H:%M:%S')}] 상태: {status}")
                
                if status in ['completed', 'failed', 'cancelled']:
                    print(f"\n작업 종료: {status}")
                    
                    if status == 'failed' and job.get('error_message'):
                        print(f"에러 메시지: {job['error_message']}")
                    
                    # 최종 로그 출력
                    print("\n최종 로그 (마지막 10줄):")
                    test_get_logs(job_id, base_url, tail=10)
                    break
                
                time.sleep(interval)
            else:
                print(f"✗ 오류: {response.status_code}")
                break
    
    except KeyboardInterrupt:
        print("\n\n모니터링 중지")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training API 테스트")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001")
    parser.add_argument("--mode", type=str, 
                       choices=["sft", "dpo", "list", "monitor"],
                       default="list")
    parser.add_argument("--job_id", type=str, help="작업 ID (monitor 모드용)")
    
    args = parser.parse_args()
    
    if args.mode == "sft":
        job_id = test_sft_training(args.base_url)
        if job_id:
            print(f"\n모니터링하려면:")
            print(f"python scripts/test_training_api.py --mode monitor --job_id {job_id}")
    
    elif args.mode == "dpo":
        job_id = test_dpo_training(args.base_url)
        if job_id:
            print(f"\n모니터링하려면:")
            print(f"python scripts/test_training_api.py --mode monitor --job_id {job_id}")
    
    elif args.mode == "list":
        test_list_jobs(args.base_url)
    
    elif args.mode == "monitor":
        if not args.job_id:
            print("오류: --job_id가 필요합니다")
            return
        monitor_job(args.job_id, args.base_url)


if __name__ == "__main__":
    main()

