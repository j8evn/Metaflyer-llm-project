import os
import json
import requests
from PIL import Image
from io import BytesIO

NAVER_CLIENT_ID = "UvhrrH5qgzxFlnYFmkTf"
NAVER_CLIENT_SECRET = "ICOBFihYe7"
API_ENDPOINT = "https://openapi.naver.com/v1/search/image"
SAVE_DIR = "data/images"
DATASET_JSON = "data/dataset.json"

# 학습할 인물 리스트 (이름: 검색어)
TARGET_PEOPLE = {
    "박나래": "박나래",
    "아이유": "아이유",
    "유재석": "유재석",
    "일론 머스크": "Elon Musk",
    "제니": "블랙핑크 제니",
    "정국": "방탄소년단 정국"
}
IMAGES_PER_PERSON = 100

def get_image_urls(query, count):
    """
    Naver 검색 API를 사용하여 이미지 URL 리스트를 반환합니다.
    """
    print(f"Fetching URLs for {query}...")
    urls = []
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    # Naver API는 한 번에 최대 100개까지만 조회 가능
    display = min(count, 100)
    
    params = {
        "query": query,
        "display": display,
        "start": 1,
        "sort": "sim", # sim: 유사도순, date: 날짜순
        "filter": "large" # 고화질 이미지 우선
    }
    
    try:
        response = requests.get(API_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'items' in data:
            for item in data['items']:
                urls.append(item['link'])
                
    except Exception as e:
        print(f"Error calling Naver API: {e}")
    
    return urls

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    new_data = []
    
    # 기존 데이터 로드 (있다면)
    if os.path.exists(DATASET_JSON):
        with open(DATASET_JSON, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                new_data.extend(existing_data)
            except:
                pass

    for name, query in TARGET_PEOPLE.items():
        person_dir = os.path.join(SAVE_DIR, query.replace(" ", "_"))
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        urls = get_image_urls(query, IMAGES_PER_PERSON)
        
        count = 0
        for i, url in enumerate(urls):
            try:
                # 이미지 다운로드
                resp = requests.get(url, timeout=10)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                
                # 파일 저장
                filename = f"{query.replace(' ', '_')}_{i:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                img.save(filepath)
                
                # 데이터셋 추가
                entry = {
                    "image": filepath,
                    "text_input": "이 인물은 누구입니까?",
                    "text_output": name
                }
                new_data.append(entry)
                count += 1
                print(f"Saved {filepath}")
                
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        print(f"Finished {name}: {count} images saved.")

    # JSON 저장
    with open(DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nDataset updated: {DATASET_JSON}")
    print(f"Total items: {len(new_data)}")

if __name__ == "__main__":
    main()
