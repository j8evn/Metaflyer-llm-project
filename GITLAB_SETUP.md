# GitLab에 프로젝트 올리기 가이드

현재 LLM 프로젝트를 GitLab에 새로운 프로젝트로 생성하고 업로드하는 완전한 가이드입니다.

## 목차
1. [사전 준비](#1-사전-준비)
2. [GitLab에서 프로젝트 생성](#2-gitlab에서-프로젝트-생성)
3. [로컬에서 Git 설정](#3-로컬에서-git-설정)
4. [코드 업로드](#4-코드-업로드)
5. [협업 설정](#5-협업-설정)

---

## 1. 사전 준비

### Git 설치 확인

```bash
# Git 버전 확인
git --version

# 설치 안 되어 있으면
# Mac: brew install git
# Ubuntu: sudo apt install git
```

### Git 설정 (처음 사용하는 경우)

```bash
# 사용자 이름 설정
git config --global user.name "Your Name"

# 이메일 설정 (GitLab 계정 이메일)
git config --global user.email "your.email@example.com"

# 설정 확인
git config --list
```

### GitLab 계정 준비

- GitLab 계정이 없다면: https://gitlab.com 에서 가입
- 이미 있다면 로그인

---

## 2. GitLab에서 프로젝트 생성

### 방법 A: 웹 인터페이스 (권장)

1. **GitLab에 로그인**
   - https://gitlab.com 접속

2. **새 프로젝트 생성**
   - 좌측 상단 `+` 버튼 클릭 → "New project/repository" 선택
   - 또는 https://gitlab.com/projects/new 직접 접속

3. **프로젝트 정보 입력**
   ```
   Project name: llm-finetuning
   Project slug: llm-finetuning (자동 생성됨)
   Visibility Level: Private (또는 Public/Internal)
   
   ✅ Initialize repository with a README (체크 해제!)
   ```

4. **Create project 클릭**

5. **프로젝트 URL 복사**
   ```
   SSH: git@gitlab.com:your-username/llm-finetuning.git
   HTTPS: https://gitlab.com/your-username/llm-finetuning.git
   ```

---

## 3. 로컬에서 Git 설정

### 현재 프로젝트 디렉토리에서

```bash
# 프로젝트 디렉토리로 이동
cd /Users/jerry/metaflyer/llm

# Git 초기화 (아직 안 했다면)
git init

# 기본 브랜치를 main으로 설정
git branch -M main
```

### .gitignore 확인

프로젝트에 이미 `.gitignore` 파일이 있습니다. 확인:

```bash
cat .gitignore
```

필요시 추가:

```bash
# 추가할 내용이 있다면
echo "추가내용" >> .gitignore
```

### 모든 파일 추가

```bash
# 모든 파일 스테이징
git add .

# 추가된 파일 확인
git status
```

### 첫 커밋 생성

```bash
# 커밋 생성
git commit -m "Initial commit: LLM Fine-tuning Project

- SFT (Supervised Fine-Tuning) 지원
- DPO (Direct Preference Optimization) 지원
- LoRA/QLoRA 파인튜닝
- FastAPI 기반 REST API 서버
- 완전한 문서 및 예제 포함"

# 커밋 확인
git log
```

---

## 4. 코드 업로드

### SSH 사용 (권장)

#### SSH 키 설정 (처음만)

```bash
# SSH 키 생성 (이미 있으면 스킵)
ssh-keygen -t ed25519 -C "your.email@example.com"
# Enter 3번 (기본 경로, 비밀번호 없음)

# 공개키 복사
cat ~/.ssh/id_ed25519.pub
# 출력된 내용 전체 복사

# GitLab에 등록
# 1. GitLab 로그인
# 2. 우측 상단 프로필 → Settings
# 3. 좌측 메뉴 → SSH Keys
# 4. 복사한 키 붙여넣기 → Add key
```

#### 원격 저장소 연결 및 푸시

```bash
# 원격 저장소 추가 (SSH)
git remote add origin git@gitlab.com:your-username/llm-finetuning.git

# 원격 저장소 확인
git remote -v

# 코드 푸시
git push -u origin main
```

### HTTPS 사용

```bash
# 원격 저장소 추가 (HTTPS)
git remote add origin https://gitlab.com/your-username/llm-finetuning.git

# 코드 푸시 (아이디/비밀번호 입력 필요)
git push -u origin main

# Personal Access Token 사용 권장
# GitLab → Settings → Access Tokens에서 생성
# Username: your-username
# Password: your-personal-access-token
```

### 푸시 성공 확인

```bash
# GitLab 웹에서 확인
# https://gitlab.com/your-username/llm-finetuning

# 또는 터미널에서
git log origin/main
```

---

## 5. 협업 설정

### README 추가 (선택)

```bash
# README가 없다면
cp README.md README.backup.md  # 백업
git add README.md
git commit -m "docs: Update README"
git push
```

### 브랜치 보호 설정

GitLab 웹에서:
1. Settings → Repository
2. Protected branches
3. main 브랜치 보호 활성화

### 협업자 추가

GitLab 웹에서:
1. Settings → Members
2. Invite members
3. 이메일 입력 + 역할 선택 (Developer/Maintainer)

### .gitlab-ci.yml 추가 (CI/CD)

```bash
cat > .gitlab-ci.yml << 'YAML'
# GitLab CI/CD 설정

stages:
  - test
  - build

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip

# Python 환경 테스트
test:python:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - python -c "import torch; print('PyTorch:', torch.__version__)"
    - python -c "import transformers; print('Transformers:', transformers.__version__)"
  only:
    - main
    - merge_requests

# 문서 빌드 (선택)
build:docs:
  stage: build
  image: python:3.10
  script:
    - echo "Documentation build complete"
  only:
    - main
YAML

git add .gitlab-ci.yml
git commit -m "ci: Add GitLab CI/CD configuration"
git push
```

---

## 빠른 시작 스크립트

전체 과정을 자동화한 스크립트:

```bash
#!/bin/bash

# 변수 설정
GITLAB_USERNAME="your-username"
PROJECT_NAME="llm-finetuning"

# 1. Git 초기화
git init
git branch -M main

# 2. 모든 파일 추가
git add .

# 3. 커밋
git commit -m "Initial commit: LLM Fine-tuning Project"

# 4. 원격 저장소 연결
git remote add origin git@gitlab.com:${GITLAB_USERNAME}/${PROJECT_NAME}.git

# 5. 푸시
git push -u origin main

echo "✅ GitLab에 코드가 업로드되었습니다!"
echo "🔗 https://gitlab.com/${GITLAB_USERNAME}/${PROJECT_NAME}"
```

---

## 단계별 체크리스트

### ✅ 시작 전
- [ ] Git 설치 확인
- [ ] Git 사용자 설정
- [ ] GitLab 계정 준비

### ✅ GitLab 설정
- [ ] 새 프로젝트 생성
- [ ] 프로젝트 URL 복사
- [ ] SSH 키 등록 (SSH 사용 시)

### ✅ 로컬 설정
- [ ] Git 초기화
- [ ] .gitignore 확인
- [ ] 파일 추가 및 커밋

### ✅ 업로드
- [ ] 원격 저장소 연결
- [ ] 코드 푸시
- [ ] GitLab 웹에서 확인

---

## 트러블슈팅

### 문제 1: SSH 연결 실패

```bash
# SSH 연결 테스트
ssh -T git@gitlab.com

# 성공 시 출력: Welcome to GitLab, @username!
```

실패 시:
- SSH 키가 GitLab에 등록되었는지 확인
- `~/.ssh/config` 설정 확인

### 문제 2: 푸시 거부 (rejected)

```bash
# 강제 푸시 (주의: 기존 내용 삭제됨)
git push -f origin main

# 또는 풀 후 푸시
git pull origin main --allow-unrelated-histories
git push origin main
```

### 문제 3: 파일 크기 제한

GitLab은 기본적으로 큰 파일을 거부합니다.

```bash
# 큰 파일 찾기
find . -type f -size +100M

# Git LFS 사용
git lfs install
git lfs track "*.bin"
git lfs track "*.pth"
git add .gitattributes
git commit -m "chore: Add Git LFS"
```

### 문제 4: 인증 실패 (HTTPS)

Personal Access Token 생성:
1. GitLab → Settings → Access Tokens
2. Token name: "llm-project"
3. Scopes: api, read_repository, write_repository
4. Create token
5. 토큰 복사하여 비밀번호 대신 사용

---

## 이후 작업 흐름

### 일반적인 워크플로우

```bash
# 1. 변경 사항 확인
git status

# 2. 파일 추가
git add .

# 3. 커밋
git commit -m "feat: Add new feature"

# 4. 푸시
git push

# 5. 풀 (다른 사람의 변경사항 가져오기)
git pull
```

### 브랜치 작업

```bash
# 새 기능 브랜치 생성
git checkout -b feature/new-feature

# 작업 후 커밋
git add .
git commit -m "feat: Implement new feature"

# 푸시
git push -u origin feature/new-feature

# GitLab에서 Merge Request 생성
```

---

## 추천 프로젝트 구조 (GitLab)

```
llm-finetuning/
├── .gitlab-ci.yml          # CI/CD 설정
├── .gitignore              # Git 무시 파일
├── README.md               # 프로젝트 설명
├── requirements.txt        # Python 의존성
├── requirements_api.txt    # API 의존성
├── LICENSE                 # 라이선스
├── CONTRIBUTING.md         # 기여 가이드
├── src/                    # 소스 코드
├── configs/                # 설정 파일
├── data/                   # 데이터 (gitignore)
├── models/                 # 모델 (gitignore)
├── outputs/                # 출력 (gitignore)
├── scripts/                # 유틸리티 스크립트
├── notebooks/              # Jupyter 노트북
└── docs/                   # 문서
```

---

## GitLab 기능 활용

### 1. Issues (이슈 관리)
```
버그 리포트, 기능 요청 등 관리
```

### 2. Merge Requests
```
코드 리뷰 및 병합
```

### 3. CI/CD Pipelines
```
자동 테스트 및 배포
```

### 4. Wiki
```
프로젝트 문서화
```

### 5. Container Registry
```
Docker 이미지 저장
```

---

## 마지막 확인

업로드 후 GitLab에서 확인:

✅ 모든 파일이 정상적으로 업로드되었는지
✅ README.md가 잘 표시되는지  
✅ .gitignore가 제대로 작동하는지 (data/, models/ 등이 제외되었는지)
✅ CI/CD 파이프라인이 실행되는지

---

## 요약

```bash
# 1. GitLab에서 프로젝트 생성
# 2. 로컬에서 실행
cd /Users/jerry/metaflyer/llm
git init
git branch -M main
git add .
git commit -m "Initial commit"
git remote add origin git@gitlab.com:username/llm-finetuning.git
git push -u origin main

# 완료! 🎉
```

프로젝트 URL: https://gitlab.com/your-username/llm-finetuning
