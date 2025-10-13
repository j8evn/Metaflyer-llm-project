"""
선호도 데이터 생성 스크립트
기존 모델의 출력을 비교하여 선호도 데이터 생성
"""

import os
import sys
import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('../src')


def generate_responses(
    model,
    tokenizer,
    prompts: list,
    num_responses: int = 2,
    temperature: float = 0.8,
    max_new_tokens: int = 256
) -> list:
    """각 프롬프트에 대해 여러 응답 생성"""
    all_responses = []
    
    for prompt in tqdm(prompts, desc="응답 생성 중"):
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        responses = []
        
        with torch.no_grad():
            for _ in range(num_responses):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 프롬프트 부분 제거
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                responses.append(response)
        
        all_responses.append(responses)
    
    return all_responses


def create_preference_dataset_from_prompts(
    prompts_file: str,
    model_path: str,
    output_file: str,
    num_responses: int = 2
):
    """
    프롬프트 파일에서 선호도 데이터셋 생성
    
    프롬프트 파일 형식 (JSON):
    [
        "프롬프트 1",
        "프롬프트 2",
        ...
    ]
    또는
    [
        {"instruction": "...", "input": "..."},
        ...
    ]
    """
    print("=" * 60)
    print("선호도 데이터셋 생성")
    print("=" * 60)
    
    # 프롬프트 로딩
    print(f"\n[1/4] 프롬프트 로딩: {prompts_file}")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # 프롬프트 포맷팅
    formatted_prompts = []
    original_prompts = []
    
    for prompt_item in prompts_data:
        if isinstance(prompt_item, dict):
            instruction = prompt_item.get('instruction', '')
            input_text = prompt_item.get('input', '')
            
            if input_text:
                formatted = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                formatted = f"""### Instruction:
{instruction}

### Response:
"""
            formatted_prompts.append(formatted)
            original_prompts.append(prompt_item)
        else:
            formatted_prompts.append(prompt_item)
            original_prompts.append(prompt_item)
    
    print(f"총 {len(formatted_prompts)}개의 프롬프트")
    
    # 모델 로딩
    print(f"\n[2/4] 모델 로딩: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    
    # 응답 생성
    print(f"\n[3/4] 각 프롬프트에 대해 {num_responses}개의 응답 생성")
    all_responses = generate_responses(
        model,
        tokenizer,
        formatted_prompts,
        num_responses=num_responses
    )
    
    # 선호도 데이터 생성 (사용자가 수동으로 선택해야 함)
    print(f"\n[4/4] 선호도 데이터 생성")
    preference_data = []
    
    for i, (prompt, responses) in enumerate(zip(original_prompts, all_responses)):
        preference_item = {
            "prompt": prompt,
            "response_1": responses[0] if len(responses) > 0 else "",
            "response_2": responses[1] if len(responses) > 1 else "",
            "chosen": "",  # 사용자가 선택해야 함
            "rejected": ""  # 사용자가 선택해야 함
        }
        preference_data.append(preference_item)
    
    # 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("선호도 데이터 생성 완료!")
    print("=" * 60)
    print(f"\n출력 파일: {output_file}")
    print("\n주의: chosen과 rejected 필드를 수동으로 채워야 합니다.")
    print("각 항목의 response_1과 response_2를 비교하여")
    print("더 좋은 응답을 'chosen'에, 덜 좋은 응답을 'rejected'에 복사하세요.")


def main():
    parser = argparse.ArgumentParser(description="선호도 데이터 생성")
    
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="프롬프트 JSON 파일 경로"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="응답 생성에 사용할 모델 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=2,
        help="각 프롬프트당 생성할 응답 수"
    )
    
    args = parser.parse_args()
    
    create_preference_dataset_from_prompts(
        prompts_file=args.prompts,
        model_path=args.model_path,
        output_file=args.output,
        num_responses=args.num_responses
    )


if __name__ == "__main__":
    main()

