"""
학습된 모델을 사용한 추론 스크립트
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """추론 엔진 클래스"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        logger.info(f"추론 디바이스: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        logger.info(f"모델 로딩 중: {self.model_path}")
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로딩
        model_kwargs = {
            'pretrained_model_name_or_path': self.model_path,
            'device_map': 'auto' if self.device == 'cuda' else None,
        }
        
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.model.eval()
        
        logger.info("모델 로딩 완료")
    
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
    ) -> list:
        """텍스트 생성"""
        
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
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """대화형 추론 (instruction format)"""
        
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
        
        results = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        
        # 응답 부분만 추출
        response = results[0]
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response


def interactive_mode(engine: InferenceEngine, config: dict):
    """대화형 모드"""
    logger.info("\n" + "=" * 50)
    logger.info("대화형 모드 시작 (종료: 'quit', 'exit', 또는 'q')")
    logger.info("=" * 50 + "\n")
    
    inference_config = config.get('inference', {})
    
    while True:
        try:
            instruction = input("\n질문을 입력하세요: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                logger.info("대화형 모드를 종료합니다.")
                break
            
            if not instruction:
                continue
            
            logger.info("\n생성 중...")
            
            response = engine.chat(
                instruction=instruction,
                max_new_tokens=inference_config.get('max_new_tokens', 256),
                temperature=inference_config.get('temperature', 0.7),
                top_p=inference_config.get('top_p', 0.9),
                top_k=inference_config.get('top_k', 50),
                repetition_penalty=inference_config.get('repetition_penalty', 1.1),
                do_sample=inference_config.get('do_sample', True)
            )
            
            print(f"\n응답:\n{response}\n")
            
        except KeyboardInterrupt:
            logger.info("\n\n대화형 모드를 종료합니다.")
            break
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="LLM 추론 스크립트")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="학습된 모델 경로"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="설정 파일 경로 (추론 설정용)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="생성할 프롬프트 (지정하지 않으면 대화형 모드)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction (instruction format 사용 시)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="추가 입력 (instruction format 사용 시)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="샘플링 온도"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="8bit 양자화로 로딩"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="4bit 양자화로 로딩"
    )
    
    args = parser.parse_args()
    
    # 설정 로딩
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 추론 엔진 초기화
    engine = InferenceEngine(
        model_path=args.model_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # 추론 모드 선택
    if args.prompt:
        # 단일 프롬프트 모드
        logger.info(f"프롬프트: {args.prompt}")
        results = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\n생성 결과:\n{results[0]}\n")
        
    elif args.instruction:
        # Instruction 모드
        logger.info(f"Instruction: {args.instruction}")
        response = engine.chat(
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\n응답:\n{response}\n")
        
    else:
        # 대화형 모드
        interactive_mode(engine, config)


if __name__ == "__main__":
    main()

