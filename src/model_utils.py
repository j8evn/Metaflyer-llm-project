"""
모델 유틸리티
모델 로딩, 설정, LoRA 적용 등의 기능 제공
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """모델 로딩 및 설정 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 가능한 디바이스: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """토크나이저 로딩"""
        logger.info(f"토크나이저 로딩: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.get('model', {}).get('trust_remote_code', False),
            cache_dir=self.config.get('model', {}).get('cache_dir', None)
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("pad_token이 없어 eos_token으로 설정했습니다")
        
        logger.info(f"토크나이저 vocab 크기: {len(tokenizer)}")
        return tokenizer
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 생성"""
        quant_config = self.config.get('quantization', {})
        
        if not quant_config.get('use_quantization', False):
            return None
        
        bits = quant_config.get('bits', 4)
        compute_dtype = quant_config.get('compute_dtype', 'float16')
        
        compute_dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }
        
        if bits == 4:
            logger.info("4-bit 양자화 설정 적용")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype_map[compute_dtype],
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            logger.info("8-bit 양자화 설정 적용")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            logger.warning(f"지원하지 않는 양자화 비트: {bits}")
            return None
    
    def load_model(self, model_name: str) -> PreTrainedModel:
        """모델 로딩"""
        logger.info(f"모델 로딩: {model_name}")
        
        quantization_config = self.get_quantization_config()
        
        # 모델 로딩 설정
        model_kwargs = {
            'pretrained_model_name_or_path': model_name,
            'trust_remote_code': self.config.get('model', {}).get('trust_remote_code', False),
            'cache_dir': self.config.get('model', {}).get('cache_dir', None),
            'device_map': 'auto' if self.device == 'cuda' else None,
        }
        
        # 양자화 설정 추가
        if quantization_config is not None:
            model_kwargs['quantization_config'] = quantization_config
        else:
            # 양자화 없을 때 dtype 설정
            advanced_config = self.config.get('advanced', {})
            if advanced_config.get('bf16', False):
                model_kwargs['torch_dtype'] = torch.bfloat16
            elif advanced_config.get('fp16', False):
                model_kwargs['torch_dtype'] = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # 그래디언트 체크포인팅 설정
        if self.config.get('advanced', {}).get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            logger.info("그래디언트 체크포인팅 활성화")
        
        logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        return model
    
    def apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """LoRA 적용"""
        lora_config = self.config.get('lora', {})
        
        if not lora_config.get('use_lora', False):
            logger.info("LoRA를 사용하지 않습니다")
            return model
        
        logger.info("LoRA 설정 적용 중...")
        
        # 양자화된 모델인 경우 준비
        if self.config.get('quantization', {}).get('use_quantization', False):
            model = prepare_model_for_kbit_training(model)
        
        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            bias=lora_config.get('bias', 'none'),
        )
        
        model = get_peft_model(model, peft_config)
        
        # 학습 가능한 파라미터 출력
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"LoRA 적용 완료")
        logger.info(f"학습 가능한 파라미터: {trainable_params / 1e6:.2f}M")
        logger.info(f"전체 파라미터: {total_params / 1e6:.2f}M")
        logger.info(f"학습 가능 비율: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def prepare_model_and_tokenizer(self, model_name: str):
        """모델과 토크나이저를 함께 준비"""
        tokenizer = self.load_tokenizer(model_name)
        model = self.load_model(model_name)
        model = self.apply_lora(model)
        
        return model, tokenizer


def print_trainable_parameters(model: PreTrainedModel):
    """학습 가능한 파라미터 정보 출력"""
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"학습 가능한 파라미터: {trainable_params / 1e6:.2f}M")
    print(f"전체 파라미터: {all_param / 1e6:.2f}M")
    print(f"학습 가능 비율: {100 * trainable_params / all_param:.2f}%")


def get_model_memory_usage():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 메모리 할당: {allocated:.2f} GB")
        print(f"GPU 메모리 예약: {reserved:.2f} GB")
    else:
        print("CUDA를 사용할 수 없습니다")

