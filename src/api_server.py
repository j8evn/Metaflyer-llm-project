"""
LLM API 서버
FastAPI를 사용한 REST API 서비스
"""

import os
import sys
import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Pydantic 모델 ==============

class GenerateRequest(BaseModel):
    """텍스트 생성 요청 모델"""
    prompt: str = Field(..., description="입력 프롬프트")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="생성할 최대 토큰 수")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="샘플링 온도")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p 샘플링")
    top_k: int = Field(50, ge=0, description="Top-k 샘플링")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="반복 패널티")
    do_sample: bool = Field(True, description="샘플링 사용 여부")
    num_return_sequences: int = Field(1, ge=1, le=5, description="생성할 응답 수")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Python이란 무엇인가요?",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }


class ChatRequest(BaseModel):
    """대화형 요청 모델 (Instruction 형식)"""
    instruction: str = Field(..., description="질문 또는 지시사항")
    input: str = Field("", description="추가 입력 (선택사항)")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Python에서 리스트를 정렬하는 방법을 알려주세요",
                "input": "",
                "temperature": 0.7
            }
        }


class GenerateResponse(BaseModel):
    """생성 응답 모델"""
    generated_text: List[str] = Field(..., description="생성된 텍스트")
    prompt: str = Field(..., description="입력 프롬프트")
    model_name: str = Field(..., description="사용된 모델 이름")
    generation_time: float = Field(..., description="생성 시간 (초)")


class ChatResponse(BaseModel):
    """대화 응답 모델"""
    response: str = Field(..., description="모델의 응답")
    instruction: str = Field(..., description="입력 질문")
    model_name: str = Field(..., description="사용된 모델 이름")
    generation_time: float = Field(..., description="생성 시간 (초)")


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    device: str


# ============== 모델 관리자 ==============

class ModelManager:
    """모델 로딩 및 관리 클래스"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
        self.model_path = None
        
    def load_model(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """모델 로딩"""
        logger.info(f"모델 로딩 시작: {model_path}")
        self.model_path = model_path
        
        try:
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로딩
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'device_map': 'auto' if self.device == 'cuda' else None,
            }
            
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info("4-bit 양자화 활성화")
            elif load_in_8bit:
                model_kwargs['load_in_8bit'] = True
                logger.info("8-bit 양자화 활성화")
            else:
                model_kwargs['torch_dtype'] = torch.float16 if self.device == 'cuda' else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.model.eval()
            
            self.model_name = os.path.basename(model_path)
            
            logger.info(f"모델 로딩 완료: {self.model_name}")
            logger.info(f"디바이스: {self.device}")
            
            if self.device == "cuda":
                logger.info(f"GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """텍스트 생성"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델이 로딩되지 않았습니다")
        
        # 입력 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성 설정
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'do_sample': do_sample,
            'num_return_sequences': num_return_sequences,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # 디코딩
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def chat(
        self,
        instruction: str,
        input_text: str = "",
        **kwargs
    ) -> str:
        """대화형 생성 (Instruction 형식)"""
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""
        
        results = self.generate(prompt, **kwargs)
        
        # 응답 부분만 추출
        response = results[0]
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def is_loaded(self) -> bool:
        """모델 로딩 상태 확인"""
        return self.model is not None and self.tokenizer is not None


# ============== 전역 모델 관리자 ==============
model_manager = ModelManager()


# ============== FastAPI 앱 설정 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    logger.info("API 서버 시작")
    
    # 환경 변수에서 모델 경로 가져오기
    model_path = os.getenv("MODEL_PATH")
    load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    
    if model_path:
        try:
            model_manager.load_model(
                model_path,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
    else:
        logger.warning("MODEL_PATH 환경 변수가 설정되지 않았습니다. /load_model 엔드포인트를 사용하세요.")
    
    yield
    
    # 종료 시
    logger.info("API 서버 종료")


app = FastAPI(
    title="LLM API 서버",
    description="오픈 소스 LLM 모델을 위한 REST API",
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
        "message": "LLM API 서버",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["기본"])
async def health_check():
    """헬스체크"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded(),
        model_name=model_manager.model_name,
        device=model_manager.device
    )


@app.post("/generate", response_model=GenerateResponse, tags=["생성"])
async def generate_text(request: GenerateRequest):
    """
    텍스트 생성 엔드포인트
    
    - **prompt**: 입력 프롬프트
    - **max_new_tokens**: 생성할 최대 토큰 수 (기본: 256)
    - **temperature**: 샘플링 온도 (기본: 0.7)
    - **top_p**: Top-p 샘플링 (기본: 0.9)
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로딩되지 않았습니다")
    
    try:
        import time
        start_time = time.time()
        
        generated_texts = model_manager.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_return_sequences=request.num_return_sequences
        )
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            generated_text=generated_texts,
            prompt=request.prompt,
            model_name=model_manager.model_name,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["생성"])
async def chat(request: ChatRequest):
    """
    대화형 생성 엔드포인트 (Instruction 형식)
    
    - **instruction**: 질문 또는 지시사항
    - **input**: 추가 입력 (선택사항)
    - **temperature**: 샘플링 온도 (기본: 0.7)
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로딩되지 않았습니다")
    
    try:
        import time
        start_time = time.time()
        
        response = model_manager.chat(
            instruction=request.instruction,
            input_text=request.input,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        generation_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            instruction=request.instruction,
            model_name=model_manager.model_name,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"대화 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model", tags=["관리"])
async def load_model(
    model_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    새로운 모델 로딩
    
    - **model_path**: 모델 경로
    - **load_in_8bit**: 8-bit 양자화 사용 여부
    - **load_in_4bit**: 4-bit 양자화 사용 여부
    """
    try:
        model_manager.load_model(
            model_path=model_path,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        return {
            "status": "success",
            "message": f"모델 로딩 완료: {model_path}",
            "model_name": model_manager.model_name,
            "device": model_manager.device
        }
        
    except Exception as e:
        logger.error(f"모델 로딩 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info", tags=["관리"])
async def model_info():
    """현재 로딩된 모델 정보"""
    if not model_manager.is_loaded():
        return {"status": "no_model_loaded"}
    
    info = {
        "model_name": model_manager.model_name,
        "model_path": model_manager.model_path,
        "device": model_manager.device,
    }
    
    if model_manager.device == "cuda":
        info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    
    return info


# ============== 메인 실행 ==============

def main():
    """서버 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM API 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")
    parser.add_argument("--model_path", type=str, help="모델 경로")
    parser.add_argument("--load_in_8bit", action="store_true", help="8-bit 양자화")
    parser.add_argument("--load_in_4bit", action="store_true", help="4-bit 양자화")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 (개발용)")
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.load_in_8bit:
        os.environ["LOAD_IN_8BIT"] = "true"
    if args.load_in_4bit:
        os.environ["LOAD_IN_4BIT"] = "true"
    
    # 서버 실행
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

