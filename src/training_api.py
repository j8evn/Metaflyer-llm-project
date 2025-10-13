"""
학습 작업을 위한 API 서버
API를 통해 SFT 및 DPO 학습을 시작하고 모니터링
"""

import os
import sys
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager
import subprocess
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Enums ==============

class TrainingType(str, Enum):
    """학습 타입"""
    SFT = "sft"
    DPO = "dpo"


class TrainingStatus(str, Enum):
    """학습 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============== Pydantic 모델 ==============

class TrainingConfig(BaseModel):
    """학습 설정"""
    model_name: str = Field(..., description="모델 이름 또는 경로")
    dataset_path: str = Field(..., description="데이터셋 경로")
    output_dir: str = Field(..., description="출력 디렉토리")
    num_epochs: int = Field(3, ge=1, le=100, description="에포크 수")
    batch_size: int = Field(4, ge=1, le=128, description="배치 크기")
    learning_rate: float = Field(2e-5, gt=0, description="학습률")
    use_lora: bool = Field(True, description="LoRA 사용 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "dataset_path": "data/train.json",
                "output_dir": "outputs/my_model",
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-5,
                "use_lora": True
            }
        }


class DPOTrainingConfig(BaseModel):
    """DPO 학습 설정"""
    model_name: str = Field(..., description="SFT 모델 경로")
    dataset_path: str = Field(..., description="선호도 데이터셋 경로")
    output_dir: str = Field(..., description="출력 디렉토리")
    num_epochs: int = Field(1, ge=1, le=10, description="에포크 수")
    batch_size: int = Field(4, ge=1, le=128, description="배치 크기")
    learning_rate: float = Field(5e-7, gt=0, description="학습률")
    beta: float = Field(0.1, gt=0, description="DPO beta 파라미터")
    use_lora: bool = Field(True, description="LoRA 사용 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "outputs/sft_model",
                "dataset_path": "data/preference_train.json",
                "output_dir": "outputs/dpo_model",
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 5e-7,
                "beta": 0.1,
                "use_lora": True
            }
        }


class TrainingJob(BaseModel):
    """학습 작업"""
    job_id: str
    training_type: TrainingType
    status: TrainingStatus
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    output_dir: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """학습 작업 응답"""
    job_id: str
    status: TrainingStatus
    message: str


class TrainingLogResponse(BaseModel):
    """학습 로그 응답"""
    job_id: str
    logs: List[str]
    total_lines: int


# ============== 작업 관리자 ==============

class TrainingJobManager:
    """학습 작업 관리 클래스"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_dir = "outputs/training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_job(
        self,
        training_type: TrainingType,
        config: Dict[str, Any]
    ) -> str:
        """새 작업 생성"""
        job_id = str(uuid.uuid4())[:8]
        
        job = TrainingJob(
            job_id=job_id,
            training_type=training_type,
            status=TrainingStatus.PENDING,
            config=config,
            created_at=datetime.now().isoformat(),
            log_file=f"{self.log_dir}/job_{job_id}.log",
            output_dir=config.get("output_dir")
        )
        
        self.jobs[job_id] = job
        logger.info(f"작업 생성됨: {job_id} ({training_type.value})")
        
        return job_id
    
    def start_job(self, job_id: str):
        """작업 시작"""
        if job_id not in self.jobs:
            raise ValueError(f"작업을 찾을 수 없습니다: {job_id}")
        
        job = self.jobs[job_id]
        
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"작업을 시작할 수 없습니다. 현재 상태: {job.status}")
        
        # 명령 구성
        if job.training_type == TrainingType.SFT:
            cmd = self._build_sft_command(job.config)
        else:
            cmd = self._build_dpo_command(job.config)
        
        # 로그 파일
        log_file = open(job.log_file, 'w')
        
        # 프로세스 시작
        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes[job_id] = process
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            
            logger.info(f"작업 시작됨: {job_id}")
            
            # 백그라운드에서 완료 모니터링
            threading.Thread(
                target=self._monitor_job,
                args=(job_id, process, log_file),
                daemon=True
            ).start()
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            log_file.close()
            logger.error(f"작업 시작 실패: {job_id} - {e}")
            raise
    
    def _monitor_job(self, job_id: str, process: subprocess.Popen, log_file):
        """작업 모니터링"""
        try:
            return_code = process.wait()
            
            job = self.jobs[job_id]
            job.completed_at = datetime.now().isoformat()
            
            if return_code == 0:
                job.status = TrainingStatus.COMPLETED
                logger.info(f"작업 완료: {job_id}")
            else:
                job.status = TrainingStatus.FAILED
                job.error_message = f"프로세스 종료 코드: {return_code}"
                logger.error(f"작업 실패: {job_id} - 코드 {return_code}")
            
        except Exception as e:
            job = self.jobs[job_id]
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"작업 모니터링 오류: {job_id} - {e}")
        
        finally:
            log_file.close()
            if job_id in self.processes:
                del self.processes[job_id]
    
    def _build_sft_command(self, config: Dict[str, Any]) -> List[str]:
        """SFT 명령 구성"""
        cmd = [
            sys.executable,
            "src/train.py",
            "--model_name", config["model_name"],
            "--dataset_path", config["dataset_path"],
            "--output_dir", config["output_dir"],
            "--num_epochs", str(config["num_epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--learning_rate", str(config["learning_rate"])
        ]
        
        if config.get("use_lora"):
            cmd.append("--use_lora")
        
        return cmd
    
    def _build_dpo_command(self, config: Dict[str, Any]) -> List[str]:
        """DPO 명령 구성"""
        cmd = [
            sys.executable,
            "src/train_dpo.py",
            "--model_name", config["model_name"],
            "--dataset_path", config["dataset_path"],
            "--output_dir", config["output_dir"],
            "--num_epochs", str(config["num_epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--learning_rate", str(config["learning_rate"]),
            "--beta", str(config["beta"])
        ]
        
        return cmd
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """작업 조회"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[TrainingJob]:
        """모든 작업 목록"""
        return list(self.jobs.values())
    
    def cancel_job(self, job_id: str):
        """작업 취소"""
        if job_id not in self.jobs:
            raise ValueError(f"작업을 찾을 수 없습니다: {job_id}")
        
        job = self.jobs[job_id]
        
        if job.status != TrainingStatus.RUNNING:
            raise ValueError(f"실행 중이 아닌 작업은 취소할 수 없습니다. 현재 상태: {job.status}")
        
        if job_id in self.processes:
            process = self.processes[job_id]
            process.terminate()
            
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            
            logger.info(f"작업 취소됨: {job_id}")
    
    def get_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """작업 로그 조회"""
        if job_id not in self.jobs:
            raise ValueError(f"작업을 찾을 수 없습니다: {job_id}")
        
        job = self.jobs[job_id]
        log_file = job.log_file
        
        if not os.path.exists(log_file):
            return []
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if tail > 0:
            lines = lines[-tail:]
        
        return [line.rstrip() for line in lines]


# ============== 전역 작업 관리자 ==============
job_manager = TrainingJobManager()


# ============== FastAPI 앱 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료"""
    logger.info("Training API 서버 시작")
    yield
    logger.info("Training API 서버 종료")


app = FastAPI(
    title="LLM Training API",
    description="API를 통한 LLM 학습 작업 관리",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== API 엔드포인트 ==============

@app.get("/", tags=["기본"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "LLM Training API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/train/sft", response_model=TrainingJobResponse, tags=["학습"])
async def start_sft_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    SFT (Supervised Fine-Tuning) 학습 시작
    
    - **model_name**: 베이스 모델 이름
    - **dataset_path**: 학습 데이터셋 경로
    - **output_dir**: 모델 저장 디렉토리
    - **num_epochs**: 에포크 수
    - **batch_size**: 배치 크기
    - **learning_rate**: 학습률
    - **use_lora**: LoRA 사용 여부
    """
    try:
        # 작업 생성
        job_id = job_manager.create_job(
            training_type=TrainingType.SFT,
            config=config.model_dump()
        )
        
        # 백그라운드에서 작업 시작
        background_tasks.add_task(job_manager.start_job, job_id)
        
        return TrainingJobResponse(
            job_id=job_id,
            status=TrainingStatus.PENDING,
            message="SFT 학습 작업이 생성되었습니다"
        )
        
    except Exception as e:
        logger.error(f"SFT 학습 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/dpo", response_model=TrainingJobResponse, tags=["학습"])
async def start_dpo_training(config: DPOTrainingConfig, background_tasks: BackgroundTasks):
    """
    DPO (Direct Preference Optimization) 학습 시작
    
    - **model_name**: SFT 모델 경로
    - **dataset_path**: 선호도 데이터셋 경로
    - **output_dir**: 모델 저장 디렉토리
    - **num_epochs**: 에포크 수
    - **batch_size**: 배치 크기
    - **learning_rate**: 학습률
    - **beta**: DPO beta 파라미터
    - **use_lora**: LoRA 사용 여부
    """
    try:
        # 작업 생성
        job_id = job_manager.create_job(
            training_type=TrainingType.DPO,
            config=config.model_dump()
        )
        
        # 백그라운드에서 작업 시작
        background_tasks.add_task(job_manager.start_job, job_id)
        
        return TrainingJobResponse(
            job_id=job_id,
            status=TrainingStatus.PENDING,
            message="DPO 학습 작업이 생성되었습니다"
        )
        
    except Exception as e:
        logger.error(f"DPO 학습 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", response_model=List[TrainingJob], tags=["작업 관리"])
async def list_jobs():
    """모든 학습 작업 목록 조회"""
    return job_manager.list_jobs()


@app.get("/jobs/{job_id}", response_model=TrainingJob, tags=["작업 관리"])
async def get_job(job_id: str):
    """특정 학습 작업 조회"""
    job = job_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    return job


@app.post("/jobs/{job_id}/cancel", tags=["작업 관리"])
async def cancel_job(job_id: str):
    """학습 작업 취소"""
    try:
        job_manager.cancel_job(job_id)
        return {"message": f"작업 {job_id}이(가) 취소되었습니다"}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"작업 취소 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/logs", response_model=TrainingLogResponse, tags=["작업 관리"])
async def get_job_logs(job_id: str, tail: int = 100):
    """
    학습 작업 로그 조회
    
    - **job_id**: 작업 ID
    - **tail**: 조회할 마지막 라인 수 (기본: 100, 0 = 전체)
    """
    try:
        logs = job_manager.get_logs(job_id, tail=tail)
        
        return TrainingLogResponse(
            job_id=job_id,
            logs=logs,
            total_lines=len(logs)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"로그 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/dataset", tags=["데이터"])
async def upload_dataset(file: UploadFile = File(...)):
    """데이터셋 업로드"""
    try:
        # 파일 저장
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"파일 업로드 완료: {file_path}")
        
        return {
            "message": "파일 업로드 성공",
            "file_path": file_path,
            "filename": file.filename,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"파일 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 메인 실행 ==============

def main():
    """서버 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Training API 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=8001, help="포트 번호")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 (개발용)")
    
    args = parser.parse_args()
    
    # 서버 실행
    uvicorn.run(
        "training_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

