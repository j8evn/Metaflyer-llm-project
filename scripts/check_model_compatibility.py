"""
모델 호환성 확인 스크립트
새로운 모델이 프로젝트와 호환되는지 확인
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import argparse


def check_model(model_name: str, quick: bool = False):
    """
    모델 호환성 체크
    
    Args:
        model_name: Hugging Face 모델 ID
        quick: 빠른 체크 (모델 로딩 스킵)
    """
    print("=" * 70)
    print(f"모델 호환성 확인: {model_name}")
    print("=" * 70)
    print()
    
    results = {
        "tokenizer": False,
        "model": False,
        "generation": False
    }
    
    try:
        # 1. 토크나이저 체크
        print("[1/4] 토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        results["tokenizer"] = True
        print(f"      ✓ 성공")
        print(f"      - Vocab 크기: {len(tokenizer):,}")
        print(f"      - 특수 토큰: {list(tokenizer.special_tokens_map.keys())}")
        
        if tokenizer.pad_token is None:
            print(f"      ⚠ pad_token이 없습니다 (자동으로 eos_token 사용)")
        
        print()
        
        if quick:
            print("[2/4] 모델 로딩... (스킵 - quick 모드)")
            print()
            print("[3/4] 파라미터 정보... (스킵 - quick 모드)")
            print()
            print("[4/4] 테스트 생성... (스킵 - quick 모드)")
            print()
        else:
            # 2. 모델 체크
            print("[2/4] 모델 로딩...")
            print("      (이 과정은 시간이 걸릴 수 있습니다...)")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            results["model"] = True
            print(f"      ✓ 성공")
            print(f"      - 모델 타입: {model.config.model_type}")
            print()
            
            # 3. 파라미터 정보
            print("[3/4] 파라미터 정보...")
            num_params = sum(p.numel() for p in model.parameters())
            print(f"      - 총 파라미터: {num_params / 1e9:.2f}B")
            print(f"      - 숨김 크기: {model.config.hidden_size}")
            print(f"      - 레이어 수: {model.config.num_hidden_layers}")
            
            if hasattr(model.config, 'max_position_embeddings'):
                print(f"      - 최대 시퀀스 길이: {model.config.max_position_embeddings}")
            
            print()
            
            # 4. 테스트 생성
            print("[4/4] 테스트 생성...")
            test_prompt = "Hello, this is a test"
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results["generation"] = True
            
            print(f"      ✓ 성공")
            print(f"      - 입력: {test_prompt}")
            print(f"      - 출력: {generated_text}")
            print()
        
        # 결과 요약
        print("=" * 70)
        if all(results.values()) or (quick and results["tokenizer"]):
            print("✅ 이 모델은 프로젝트와 호환됩니다!")
            print()
            print("사용 방법:")
            print(f'  python src/train.py --model_name "{model_name}"')
            print()
            
            # 추천 설정
            print("추천 설정:")
            if "qwen" in model_name.lower() or "falcon" in model_name.lower():
                print("  ⚠ 이 모델은 trust_remote_code=True가 필요합니다")
                print("  configs/train_config.yaml에서 설정하세요:")
                print("    model:")
                print("      trust_remote_code: true")
            
            if "gemma" in model_name.lower():
                print("  ⚠ Gemma는 Hugging Face 인증이 필요합니다")
                print("  먼저 로그인하세요: huggingface-cli login")
            
            return True
        else:
            print("❌ 일부 테스트가 실패했습니다")
            print()
            print("실패한 항목:")
            for key, value in results.items():
                if not value:
                    print(f"  - {key}")
            return False
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 오류 발생: {type(e).__name__}")
        print(f"   {str(e)}")
        print()
        
        # 일반적인 오류 해결 방법
        error_msg = str(e).lower()
        
        if "401" in error_msg or "authentication" in error_msg:
            print("해결 방법:")
            print("  1. Hugging Face 계정으로 로그인하세요:")
            print("     huggingface-cli login")
            print("  2. 모델 접근 권한을 확인하세요:")
            print(f"     https://huggingface.co/{model_name}")
        
        elif "trust_remote_code" in error_msg:
            print("해결 방법:")
            print("  configs/train_config.yaml에 다음을 추가하세요:")
            print("    model:")
            print("      trust_remote_code: true")
        
        elif "memory" in error_msg or "oom" in error_msg:
            print("해결 방법:")
            print("  --quick 플래그를 사용하여 빠른 체크만 수행하세요:")
            print(f"  python scripts/check_model_compatibility.py '{model_name}' --quick")
        
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="모델 호환성 확인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 체크
  python scripts/check_model_compatibility.py "mistralai/Mistral-7B-v0.1"
  
  # 빠른 체크 (모델 로딩 스킵)
  python scripts/check_model_compatibility.py "meta-llama/Llama-2-7b-hf" --quick
  
  # 여러 모델 체크
  python scripts/check_model_compatibility.py "gpt2" --quick
  python scripts/check_model_compatibility.py "google/gemma-7b" --quick
        """
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        help="확인할 Hugging Face 모델 ID"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 체크 (토크나이저만 확인, 모델 로딩 스킵)"
    )
    
    args = parser.parse_args()
    
    success = check_model(args.model_name, quick=args.quick)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

