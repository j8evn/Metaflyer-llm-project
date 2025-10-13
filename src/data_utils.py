"""
데이터 처리 유틸리티
학습 데이터 로딩, 전처리, 포맷팅 기능 제공
"""

import json
import os
from typing import List, Dict, Optional, Union
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """데이터셋 로딩 및 전처리 클래스"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        train_split: float = 0.9
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_split = train_split
    
    def load_from_json(self, file_path: str) -> Dataset:
        """JSON 파일에서 데이터셋 로딩"""
        logger.info(f"JSON 파일에서 데이터 로딩: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON 데이터는 리스트 형식이어야 합니다")
        
        logger.info(f"로딩된 샘플 수: {len(data)}")
        return Dataset.from_list(data)
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train") -> Dataset:
        """Hugging Face 데이터셋 허브에서 로딩"""
        logger.info(f"Hugging Face에서 데이터셋 로딩: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    
    def format_instruction_dataset(self, examples: Dict) -> Dict:
        """
        instruction-input-output 형식의 데이터를 프롬프트로 변환
        
        Format:
        ### Instruction:
        {instruction}
        
        ### Input:
        {input}
        
        ### Response:
        {output}
        """
        prompts = []
        
        for i in range(len(examples.get('instruction', []))):
            instruction = examples['instruction'][i]
            input_text = examples.get('input', [''] * len(examples['instruction']))[i]
            output_text = examples['output'][i]
            
            if input_text:
                prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
            else:
                prompt = f"""### Instruction:
{instruction}

### Response:
{output_text}"""
            
            prompts.append(prompt)
        
        return {"text": prompts}
    
    def format_chat_dataset(self, examples: Dict) -> Dict:
        """
        대화형 데이터셋 포맷팅 (messages 형식)
        """
        formatted_texts = []
        
        for messages in examples.get('messages', []):
            text = ""
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                text += f"<|{role}|>\n{content}\n\n"
            formatted_texts.append(text.strip())
        
        return {"text": formatted_texts}
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """텍스트 토크나이징"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Labels는 input_ids와 동일하게 설정 (Causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        format_type: str = "instruction",
        remove_columns: Optional[List[str]] = None
    ) -> Dataset:
        """
        데이터셋 전처리 및 토크나이징
        
        Args:
            dataset: 원본 데이터셋
            format_type: 포맷 타입 ("instruction", "chat", "text")
            remove_columns: 제거할 컬럼 리스트
        """
        logger.info(f"데이터셋 전처리 시작 (format: {format_type})")
        
        # 포맷팅
        if format_type == "instruction":
            dataset = dataset.map(
                self.format_instruction_dataset,
                batched=True,
                remove_columns=remove_columns or dataset.column_names
            )
        elif format_type == "chat":
            dataset = dataset.map(
                self.format_chat_dataset,
                batched=True,
                remove_columns=remove_columns or dataset.column_names
            )
        elif format_type == "text":
            # 이미 "text" 컬럼이 있다고 가정
            if "text" not in dataset.column_names:
                raise ValueError("'text' 컬럼이 데이터셋에 없습니다")
        
        # 토크나이징
        logger.info("토크나이징 진행 중...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"전처리 완료. 샘플 수: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset) -> tuple:
        """데이터셋을 학습/검증 세트로 분할"""
        if self.train_split >= 1.0:
            logger.warning("train_split이 1.0이므로 검증 세트를 생성하지 않습니다")
            return dataset, None
        
        split_dataset = dataset.train_test_split(
            test_size=1.0 - self.train_split,
            seed=42
        )
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        logger.info(f"학습 세트: {len(train_dataset)} 샘플")
        logger.info(f"검증 세트: {len(eval_dataset)} 샘플")
        
        return train_dataset, eval_dataset


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    예제 데이터셋 생성 (테스트용)
    """
    sample_data = [
        {
            "instruction": "다음 문장을 한국어로 번역하세요.",
            "input": "Hello, how are you?",
            "output": "안녕하세요, 어떻게 지내세요?"
        },
        {
            "instruction": "다음 질문에 답하세요.",
            "input": "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
            "output": "리스트는 변경 가능한(mutable) 자료구조이고, 튜플은 변경 불가능한(immutable) 자료구조입니다. 리스트는 대괄호 []를 사용하고, 튜플은 소괄호 ()를 사용합니다."
        },
        {
            "instruction": "다음 코드의 결과를 예측하세요.",
            "input": "x = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
            "output": "[1, 2, 3, 4]가 출력됩니다. 리스트는 참조로 전달되므로 y를 수정하면 x도 함께 변경됩니다."
        },
        {
            "instruction": "간단한 자기소개를 작성하세요.",
            "input": "",
            "output": "안녕하세요! 저는 다양한 주제에 대해 도움을 드릴 수 있는 AI 어시스턴트입니다. 질문이나 요청사항이 있으시면 언제든지 말씀해 주세요."
        }
    ]
    
    # 샘플 반복하여 요청된 수만큼 생성
    full_data = []
    for i in range(num_samples):
        sample = sample_data[i % len(sample_data)].copy()
        full_data.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"샘플 데이터셋 생성 완료: {output_path} ({num_samples} 샘플)")


if __name__ == "__main__":
    # 샘플 데이터셋 생성 예제
    create_sample_dataset("../data/train.json", num_samples=100)
    create_sample_dataset("../data/eval.json", num_samples=20)
    print("샘플 데이터셋이 생성되었습니다!")

