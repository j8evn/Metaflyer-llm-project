"""
DPO (Direct Preference Optimization) 유틸리티
선호도 데이터 처리 및 DPO 학습 관련 기능
"""

import json
import os
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceDatasetLoader:
    """선호도 데이터셋 로딩 및 전처리 클래스"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def load_from_json(self, file_path: str) -> Dataset:
        """JSON 파일에서 선호도 데이터셋 로딩"""
        logger.info(f"선호도 데이터 로딩: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON 데이터는 리스트 형식이어야 합니다")
        
        # 데이터 검증
        required_keys = ['prompt', 'chosen', 'rejected']
        for i, item in enumerate(data):
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"샘플 {i}에 '{key}' 키가 없습니다")
        
        logger.info(f"로딩된 선호도 샘플 수: {len(data)}")
        return Dataset.from_list(data)
    
    def format_preference_data(self, examples: Dict) -> Dict:
        """
        선호도 데이터를 DPO 형식으로 변환
        
        입력 형식:
        {
            "prompt": "질문 또는 프롬프트",
            "chosen": "선호하는 응답",
            "rejected": "선호하지 않는 응답"
        }
        """
        formatted_data = {
            'prompt': [],
            'chosen': [],
            'rejected': []
        }
        
        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i]
            chosen = examples['chosen'][i]
            rejected = examples['rejected'][i]
            
            # Instruction 형식으로 프롬프트 구성
            if isinstance(prompt, dict):
                # instruction-input 형식
                instruction = prompt.get('instruction', '')
                input_text = prompt.get('input', '')
                
                if input_text:
                    formatted_prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
                else:
                    formatted_prompt = f"""### Instruction:
{instruction}

### Response:
"""
            else:
                # 단순 텍스트 프롬프트
                formatted_prompt = prompt
            
            formatted_data['prompt'].append(formatted_prompt)
            formatted_data['chosen'].append(chosen)
            formatted_data['rejected'].append(rejected)
        
        return formatted_data
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        format_prompts: bool = True
    ) -> Dataset:
        """
        DPO 학습을 위한 데이터셋 전처리
        
        Args:
            dataset: 원본 데이터셋
            format_prompts: 프롬프트 포맷팅 여부
        """
        logger.info("DPO 데이터셋 전처리 시작")
        
        if format_prompts:
            # 프롬프트가 dict 형식인 경우 포맷팅
            first_prompt = dataset[0]['prompt']
            if isinstance(first_prompt, dict):
                logger.info("프롬프트 포맷팅 적용")
                dataset = dataset.map(
                    self.format_preference_data,
                    batched=True
                )
        
        logger.info(f"전처리 완료. 샘플 수: {len(dataset)}")
        return dataset


def create_preference_sample_dataset(output_path: str, num_samples: int = 50):
    """
    샘플 선호도 데이터셋 생성 (테스트용)
    
    선호도 데이터는 다음 형식을 따릅니다:
    - prompt: 질문 또는 지시사항
    - chosen: 더 좋은 응답 (선호)
    - rejected: 덜 좋은 응답 (비선호)
    """
    sample_data = [
        {
            "prompt": "Python에서 리스트와 튜플의 차이점을 설명하세요.",
            "chosen": "리스트는 변경 가능한(mutable) 자료구조로, 요소를 추가, 삭제, 수정할 수 있습니다. 대괄호 []로 표현하며, 예를 들어 [1, 2, 3]과 같이 사용합니다. 반면 튜플은 변경 불가능한(immutable) 자료구조로, 한번 생성되면 요소를 수정할 수 없습니다. 소괄호 ()로 표현하며, (1, 2, 3)과 같이 사용합니다. 튜플은 불변성으로 인해 딕셔너리의 키로 사용할 수 있고, 메모리 효율성도 더 좋습니다.",
            "rejected": "리스트는 []를 사용하고 튜플은 ()를 사용합니다. 리스트는 변경할 수 있고 튜플은 변경할 수 없습니다."
        },
        {
            "prompt": "머신러닝과 딥러닝의 차이점은 무엇인가요?",
            "chosen": "머신러닝은 컴퓨터가 데이터로부터 패턴을 학습하여 예측하는 인공지능의 한 분야입니다. 의사결정 트리, 랜덤 포레스트, SVM 등 다양한 알고리즘을 포함합니다. 딥러닝은 머신러닝의 하위 분야로, 인공 신경망, 특히 깊은 신경망(여러 계층)을 사용하여 복잡한 패턴을 학습합니다. 딥러닝은 이미지 인식, 자연어 처리 등에서 뛰어난 성능을 보이지만, 더 많은 데이터와 컴퓨팅 리소스가 필요합니다.",
            "rejected": "머신러닝은 일반적인 AI이고, 딥러닝은 신경망을 사용하는 것입니다."
        },
        {
            "prompt": "다음 코드를 설명하세요: list(map(lambda x: x**2, [1, 2, 3]))",
            "chosen": "이 코드는 리스트의 각 요소를 제곱하는 함수형 프로그래밍 표현입니다. 단계별로 살펴보면:\n1. lambda x: x**2는 입력값을 제곱하는 익명 함수입니다\n2. map() 함수는 이 람다 함수를 [1, 2, 3] 리스트의 모든 요소에 적용합니다\n3. list()로 변환하여 최종 결과 [1, 4, 9]를 얻습니다\n\n동일한 결과를 리스트 컴프리헨션으로 표현하면 [x**2 for x in [1, 2, 3]]입니다.",
            "rejected": "리스트의 숫자들을 제곱하는 코드입니다. 결과는 [1, 4, 9]입니다."
        },
        {
            "prompt": "RESTful API란 무엇인가요?",
            "chosen": "RESTful API는 REST(Representational State Transfer) 아키텍처 원칙을 따르는 웹 API입니다. 주요 특징은:\n\n1. 자원(Resource) 기반: URL로 자원을 식별합니다 (예: /users/123)\n2. HTTP 메서드 사용: GET(조회), POST(생성), PUT(수정), DELETE(삭제)\n3. 무상태(Stateless): 각 요청은 독립적이며 서버가 세션 상태를 저장하지 않습니다\n4. 표준 형식: JSON이나 XML로 데이터를 주고받습니다\n\n예시: GET /api/users는 사용자 목록을 조회하고, POST /api/users는 새 사용자를 생성합니다.",
            "rejected": "RESTful API는 웹에서 데이터를 주고받는 방법입니다. HTTP를 사용합니다."
        },
        {
            "prompt": "데이터베이스 정규화가 왜 중요한가요?",
            "chosen": "데이터베이스 정규화는 데이터 중복을 최소화하고 데이터 무결성을 보장하기 위한 설계 과정입니다. 중요한 이유는:\n\n1. 데이터 중복 제거: 같은 정보가 여러 곳에 저장되는 것을 방지합니다\n2. 갱신 이상 방지: 데이터 수정 시 일관성을 유지합니다\n3. 저장 공간 절약: 중복 데이터가 없어 효율적입니다\n4. 쿼리 성능 향상: 잘 정규화된 스키마는 조인 최적화가 가능합니다\n\n단, 과도한 정규화는 조인이 많아져 성능이 저하될 수 있으므로, 상황에 따라 역정규화도 고려해야 합니다.",
            "rejected": "데이터 중복을 줄이고 데이터를 효율적으로 저장하기 위해서입니다."
        },
        {
            "prompt": "Git과 GitHub의 차이는 무엇인가요?",
            "chosen": "Git과 GitHub는 서로 다른 개념입니다:\n\nGit:\n- 버전 관리 시스템(VCS)\n- 로컬 컴퓨터에서 실행되는 소프트웨어\n- 코드 변경 이력을 추적하고 관리\n- 명령줄 또는 GUI 도구로 사용\n- 2005년 리누스 토발즈가 개발\n\nGitHub:\n- Git 저장소를 호스팅하는 웹 기반 플랫폼\n- 협업 기능 제공 (Pull Request, Issues, Projects)\n- 소셜 코딩 기능 (팔로우, 스타)\n- CI/CD, GitHub Actions 등 추가 서비스\n- 대안: GitLab, Bitbucket\n\n요약하면, Git은 도구이고 GitHub는 Git을 사용하는 서비스입니다.",
            "rejected": "Git은 버전 관리 프로그램이고 GitHub는 Git을 사용하는 웹사이트입니다."
        },
        {
            "prompt": {
                "instruction": "다음 문장을 영어로 번역하세요.",
                "input": "오늘 날씨가 정말 좋네요."
            },
            "chosen": "The weather is really nice today.",
            "rejected": "Today weather really good."
        },
        {
            "prompt": {
                "instruction": "다음 질문에 답하세요.",
                "input": "인공지능이 인류에게 미칠 영향은?"
            },
            "chosen": "인공지능은 인류에게 긍정적, 부정적 영향을 모두 미칠 수 있습니다.\n\n긍정적 영향:\n- 의료 진단 및 치료 개선\n- 생산성 향상 및 자동화\n- 과학 연구 가속화\n- 개인화된 교육 및 서비스\n\n부정적 우려:\n- 일자리 대체 및 경제적 불평등\n- 프라이버시 침해 위험\n- 편향성 및 차별 가능성\n- 안전성 및 통제 문제\n\n핵심은 책임감 있는 AI 개발과 적절한 규제를 통해 혜택을 극대화하고 위험을 최소화하는 것입니다.",
            "rejected": "인공지능은 좋을 수도 나쁠 수도 있습니다. 일자리를 없앨 수도 있고 편리하게 만들 수도 있습니다."
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
    
    logger.info(f"선호도 데이터셋 생성 완료: {output_path} ({num_samples} 샘플)")


if __name__ == "__main__":
    # 샘플 선호도 데이터셋 생성
    create_preference_sample_dataset("../data/preference_train.json", num_samples=50)
    create_preference_sample_dataset("../data/preference_eval.json", num_samples=10)
    print("샘플 선호도 데이터셋이 생성되었습니다!")

