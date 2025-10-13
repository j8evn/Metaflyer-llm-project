"""
학습된 모델 평가 스크립트
"""

import os
import sys
import argparse
import json
import yaml
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('../src')


def load_model_and_tokenizer(model_path: str):
    """모델과 토크나이저 로딩"""
    print(f"모델 로딩: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """응답 생성"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def evaluate_on_dataset(model, tokenizer, eval_data: list, max_samples: int = None):
    """데이터셋에 대한 평가"""
    results = []
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"\n평가 시작 ({len(eval_data)} 샘플)")
    
    for i, sample in enumerate(tqdm(eval_data)):
        # Instruction 형식으로 프롬프트 구성
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        expected_output = sample.get('output', '')
        
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
        
        # 응답 생성
        generated = generate_response(model, tokenizer, prompt)
        
        # 결과 저장
        result = {
            'index': i,
            'instruction': instruction,
            'input': input_text,
            'expected': expected_output,
            'generated': generated,
            'prompt': prompt
        }
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="모델 평가 스크립트")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="평가할 모델 경로"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="평가 데이터 경로 (JSON)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="평가 결과 저장 경로"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="평가할 최대 샘플 수"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수"
    )
    
    args = parser.parse_args()
    
    # 모델 로딩
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 평가 데이터 로딩
    print(f"\n평가 데이터 로딩: {args.eval_data}")
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 평가 실행
    results = evaluate_on_dataset(
        model,
        tokenizer,
        eval_data,
        max_samples=args.max_samples
    )
    
    # 결과 저장
    print(f"\n결과 저장: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 몇 가지 예시 출력
    print("\n" + "=" * 80)
    print("평가 예시 (처음 3개):")
    print("=" * 80)
    
    for result in results[:3]:
        print(f"\n[샘플 {result['index']}]")
        print(f"Instruction: {result['instruction']}")
        if result['input']:
            print(f"Input: {result['input']}")
        print(f"\nExpected: {result['expected']}")
        print(f"\nGenerated:\n{result['generated']}")
        print("-" * 80)
    
    print(f"\n전체 결과가 {args.output_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()

