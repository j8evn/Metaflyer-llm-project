# 이미지 디렉토리

멀티모달 학습을 위한 이미지 파일들을 이 디렉토리에 저장하세요.

## 지원 형식

- JPG/JPEG
- PNG
- GIF
- BMP
- WEBP

## 이미지 요구사항

- **최소 해상도**: 224x224 픽셀
- **권장 해상도**: 512x512 이상
- **최대 크기**: 10MB 이하 권장
- **형식**: RGB (컬러) 또는 Grayscale

## 디렉토리 구조 예시

```
data/images/
├── cat.jpg
├── dog.jpg
├── food.jpg
├── landscape.jpg
├── city.jpg
└── ...
```

## 샘플 이미지 다운로드

공개 데이터셋이나 무료 이미지 사이트에서 다운로드:

- Unsplash: https://unsplash.com/
- Pexels: https://www.pexels.com/
- COCO Dataset: https://cocodataset.org/

## 사용 방법

1. 이미지를 이 디렉토리에 복사
2. `data/multimodal_train.json`에서 이미지 경로 지정
3. 학습 실행

```bash
python src/train_multimodal.py --config configs/multimodal_config.yaml
```
