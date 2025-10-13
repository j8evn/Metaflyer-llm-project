#!/bin/bash

# GitLab 업로드 스크립트
# 사용법: ./gitlab_upload.sh your-gitlab-username project-name

set -e  # 오류 발생 시 중단

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GitLab 프로젝트 업로드 스크립트${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 인자 확인
if [ $# -lt 2 ]; then
    echo -e "${RED}사용법: ./gitlab_upload.sh <gitlab-username> <project-name>${NC}"
    echo ""
    echo "예제:"
    echo "  ./gitlab_upload.sh myusername llm-finetuning"
    echo ""
    exit 1
fi

GITLAB_USERNAME=$1
PROJECT_NAME=$2
USE_SSH=${3:-"yes"}  # 기본값: SSH 사용

echo -e "${YELLOW}설정:${NC}"
echo "  GitLab 사용자: $GITLAB_USERNAME"
echo "  프로젝트 이름: $PROJECT_NAME"
echo "  연결 방식: $([ "$USE_SSH" = "yes" ] && echo "SSH" || echo "HTTPS")"
echo ""

# Git 설치 확인
echo -e "${YELLOW}[1/7] Git 설치 확인...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git이 설치되지 않았습니다.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git 설치됨: $(git --version)${NC}"
echo ""

# Git 사용자 설정 확인
echo -e "${YELLOW}[2/7] Git 설정 확인...${NC}"
if [ -z "$(git config user.name)" ]; then
    echo -e "${YELLOW}Git 사용자 이름을 입력하세요:${NC}"
    read -p "이름: " git_name
    git config --global user.name "$git_name"
fi

if [ -z "$(git config user.email)" ]; then
    echo -e "${YELLOW}Git 이메일을 입력하세요:${NC}"
    read -p "이메일: " git_email
    git config --global user.email "$git_email"
fi

echo -e "${GREEN}✓ Git 사용자: $(git config user.name) <$(git config user.email)>${NC}"
echo ""

# Git 저장소 초기화
echo -e "${YELLOW}[3/7] Git 저장소 초기화...${NC}"
if [ -d ".git" ]; then
    echo -e "${YELLOW}기존 Git 저장소가 있습니다. 덮어쓰시겠습니까? (y/n)${NC}"
    read -p "> " overwrite
    if [ "$overwrite" = "y" ]; then
        rm -rf .git
        git init
        git branch -M main
    fi
else
    git init
    git branch -M main
fi
echo -e "${GREEN}✓ Git 저장소 초기화 완료${NC}"
echo ""

# .gitignore 확인
echo -e "${YELLOW}[4/7] .gitignore 확인...${NC}"
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}.gitignore 파일을 생성합니다...${NC}"
    cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
*.egg-info/

# Data & Models
data/*.json
data/*.csv
models/
outputs/
*.bin
*.safetensors
*.pth

# Logs
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
GITIGNORE
fi
echo -e "${GREEN}✓ .gitignore 확인 완료${NC}"
echo ""

# 파일 추가 및 커밋
echo -e "${YELLOW}[5/7] 파일 추가 및 커밋...${NC}"
git add .

# 커밋 메시지
COMMIT_MSG="Initial commit: LLM Fine-tuning Project

- SFT (Supervised Fine-Tuning) 지원
- DPO (Direct Preference Optimization) 지원  
- LoRA/QLoRA 파인튜닝
- FastAPI 기반 REST API 서버
- 완전한 문서 및 예제 포함"

git commit -m "$COMMIT_MSG"
echo -e "${GREEN}✓ 커밋 완료${NC}"
echo ""

# 원격 저장소 URL 구성
if [ "$USE_SSH" = "yes" ]; then
    REMOTE_URL="git@gitlab.com:${GITLAB_USERNAME}/${PROJECT_NAME}.git"
else
    REMOTE_URL="https://gitlab.com/${GITLAB_USERNAME}/${PROJECT_NAME}.git"
fi

# 원격 저장소 연결
echo -e "${YELLOW}[6/7] 원격 저장소 연결...${NC}"
echo "URL: $REMOTE_URL"

# 기존 origin 제거
git remote remove origin 2>/dev/null || true

git remote add origin "$REMOTE_URL"
echo -e "${GREEN}✓ 원격 저장소 연결 완료${NC}"
echo ""

# 푸시
echo -e "${YELLOW}[7/7] GitLab에 코드 푸시...${NC}"
echo ""

if [ "$USE_SSH" = "yes" ]; then
    # SSH 연결 테스트
    echo "SSH 연결 테스트 중..."
    if ssh -T git@gitlab.com 2>&1 | grep -q "Welcome to GitLab"; then
        echo -e "${GREEN}✓ SSH 연결 성공${NC}"
    else
        echo -e "${RED}✗ SSH 연결 실패${NC}"
        echo ""
        echo "SSH 키를 GitLab에 등록해야 합니다:"
        echo "1. SSH 키 생성: ssh-keygen -t ed25519 -C \"your@email.com\""
        echo "2. 공개키 복사: cat ~/.ssh/id_ed25519.pub"
        echo "3. GitLab → Settings → SSH Keys에 등록"
        echo ""
        echo "또는 HTTPS를 사용하려면:"
        echo "./gitlab_upload.sh $GITLAB_USERNAME $PROJECT_NAME no"
        exit 1
    fi
fi

# 푸시 실행
echo "푸시 시작..."
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ GitLab 업로드 완료!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "프로젝트 URL:"
    echo -e "${GREEN}🔗 https://gitlab.com/${GITLAB_USERNAME}/${PROJECT_NAME}${NC}"
    echo ""
    echo "다음 단계:"
    echo "1. 웹 브라우저에서 프로젝트 확인"
    echo "2. Settings → General → Visibility 설정"
    echo "3. Settings → Members에서 협업자 추가"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ 푸시 실패${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "가능한 원인:"
    echo "1. GitLab에 프로젝트가 생성되지 않았습니다"
    echo "   → https://gitlab.com/projects/new 에서 프로젝트 생성"
    echo ""
    echo "2. 인증 실패 (HTTPS 사용 시)"
    echo "   → Personal Access Token 사용"
    echo "   → GitLab → Settings → Access Tokens"
    echo ""
    echo "3. 권한 문제"
    echo "   → 프로젝트에 대한 쓰기 권한 확인"
    echo ""
    exit 1
fi
