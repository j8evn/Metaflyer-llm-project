"""
DPO (Direct Preference Optimization) 학습 스크립트
선호도 데이터를 사용한 RLHF 대안
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from typing import Optional

import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

from model_utils import ModelLoader, print_trainable_parameters, get_model_memory_usage
from dpo_utils import PreferenceDatasetLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'dpo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로딩"""
    logger.info(f"설정 파일 로딩: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """커맨드 라인 인자와 설정 파일 병합"""
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_path:
        config['data']['train_path'] = args.dataset_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    return config


def create_dpo_config(config: dict) -> DPOConfig:
    """DPO 학습 설정 생성"""
    training_config = config['training']
    advanced_config = config.get('advanced', {})
    monitoring_config = config.get('monitoring', {})
    dpo_config_dict = config.get('dpo', {})
    
    args = DPOConfig(
        # 기본 학습 설정
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        
        # DPO 특화 설정
        beta=dpo_config_dict.get('beta', 0.1),  # DPO 손실 함수의 온도 파라미터
        loss_type=dpo_config_dict.get('loss_type', 'sigmoid'),  # sigmoid, hinge, ipo, kto_pair
        
        # 저장 및 평가 설정
        save_strategy="steps",
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        evaluation_strategy="steps" if 'eval_path' in config['data'] else "no",
        eval_steps=training_config.get('eval_steps', 500),
        
        # 로깅 설정
        logging_dir=f"{training_config['output_dir']}/logs",
        logging_steps=training_config['logging_steps'],
        report_to=monitoring_config.get('report_to', ['tensorboard']),
        
        # 최적화 설정
        optim=config.get('optimizer', {}).get('name', 'adamw_torch'),
        lr_scheduler_type=config.get('optimizer', {}).get('lr_scheduler', 'cosine'),
        
        # 혼합 정밀도 설정
        fp16=advanced_config.get('fp16', False),
        bf16=advanced_config.get('bf16', True),
        
        # 기타 설정
        dataloader_num_workers=advanced_config.get('dataloader_num_workers', 4),
        remove_unused_columns=False,  # DPO에서는 필수
        
        # 시드
        seed=42,
        
        # 메타데이터
        run_name=f"dpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    return args


def train_dpo(config: dict):
    """DPO 메인 학습 함수"""
    logger.info("=" * 50)
    logger.info("DPO (Direct Preference Optimization) 학습 시작")
    logger.info("=" * 50)
    
    # 모델과 토크나이저 로딩
    model_name = config['model']['name']
    logger.info(f"\n단계 1: 모델 및 토크나이저 로딩 - {model_name}")
    
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.prepare_model_and_tokenizer(model_name)
    
    # Reference 모델 로딩 (DPO는 참조 모델 필요)
    logger.info("참조 모델 로딩 중...")
    ref_model = None
    if not config.get('dpo', {}).get('use_peft_for_reference', False):
        # 별도의 참조 모델 로딩
        ref_model = model_loader.load_model(model_name)
        ref_model.eval()
        logger.info("별도의 참조 모델 로딩 완료")
    else:
        logger.info("PEFT 사용 시 참조 모델은 자동으로 처리됩니다")
    
    print_trainable_parameters(model)
    get_model_memory_usage()
    
    # 선호도 데이터셋 로딩 및 전처리
    logger.info(f"\n단계 2: 선호도 데이터셋 로딩 및 전처리")
    
    data_config = config['data']
    dataset_loader = PreferenceDatasetLoader(
        tokenizer=tokenizer,
        max_length=data_config.get('max_length', 512),
        max_prompt_length=data_config.get('max_prompt_length', 256)
    )
    
    # 학습 데이터 로딩
    train_dataset = dataset_loader.load_from_json(data_config['train_path'])
    train_dataset = dataset_loader.prepare_dataset(train_dataset)
    
    # 검증 데이터셋 로딩 (있는 경우)
    eval_dataset = None
    if 'eval_path' in data_config and os.path.exists(data_config['eval_path']):
        eval_dataset = dataset_loader.load_from_json(data_config['eval_path'])
        eval_dataset = dataset_loader.prepare_dataset(eval_dataset)
        logger.info(f"검증 데이터: {len(eval_dataset)} 샘플")
    
    # DPO 설정 생성
    logger.info(f"\n단계 3: DPO 학습 설정")
    dpo_config_obj = create_dpo_config(config)
    
    # DPO Trainer 생성
    logger.info(f"\n단계 4: DPO Trainer 초기화")
    
    dpo_config_params = config.get('dpo', {})
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config_obj,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=dpo_config_params.get('beta', 0.1),
        max_length=data_config.get('max_length', 512),
        max_prompt_length=data_config.get('max_prompt_length', 256),
    )
    
    # 학습 시작
    logger.info(f"\n단계 5: DPO 학습 시작")
    logger.info("=" * 50)
    
    try:
        trainer.train()
        logger.info("DPO 학습 완료!")
        
        # 모델 저장
        output_dir = config['training']['output_dir']
        final_output_dir = f"{output_dir}/final_model"
        logger.info(f"\n모델 저장 중: {final_output_dir}")
        
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        logger.info("모델 저장 완료!")
        logger.info(f"모델 경로: {final_output_dir}")
        
        # 최종 메모리 사용량
        get_model_memory_usage()
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}", exc_info=True)
        raise
    
    logger.info("=" * 50)
    logger.info("DPO 학습 완료")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="DPO 학습 스크립트")
    
    # 설정 파일 또는 개별 인자
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="설정 파일 경로 (YAML)"
    )
    
    # 개별 인자 (설정 파일보다 우선)
    parser.add_argument("--model_name", type=str, help="모델 이름 (Hugging Face ID)")
    parser.add_argument("--dataset_path", type=str, help="선호도 데이터셋 경로")
    parser.add_argument("--output_dir", type=str, help="출력 디렉토리")
    parser.add_argument("--num_epochs", type=int, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, help="학습률")
    parser.add_argument("--beta", type=float, help="DPO beta 파라미터")
    
    args = parser.parse_args()
    
    # 설정 로딩
    if not os.path.exists(args.config):
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # 커맨드 라인 인자와 병합
    config = merge_args_with_config(args, config)
    
    # beta 파라미터 병합
    if args.beta:
        if 'dpo' not in config:
            config['dpo'] = {}
        config['dpo']['beta'] = args.beta
    
    # 출력 디렉토리 생성
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # 설정 출력
    logger.info("사용 중인 설정:")
    logger.info(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    
    # DPO 학습 시작
    train_dpo(config)


if __name__ == "__main__":
    main()

