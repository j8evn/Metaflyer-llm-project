import os
import json
import requests
from PIL import Image
from io import BytesIO

NAVER_CLIENT_ID = "UvhrrH5qgzxFlnYFmkTf"
NAVER_CLIENT_SECRET = "ICOBFihYe7"
API_ENDPOINT = "https://openapi.naver.com/v1/search/image"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "data/train")
PEOPLE_LIST_JSON = os.path.join(BASE_DIR, "config/people.json")
DATASET_JSON = os.path.join(BASE_DIR, "data/dataset.json")
IMAGES_PER_PERSON = 30 

def load_target_people():
    if os.path.exists(PEOPLE_LIST_JSON):
        with open(PEOPLE_LIST_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Warning: {PEOPLE_LIST_JSON} not found. Using default empty list.")
    return {}

TARGET_PEOPLE = load_target_people()

def get_image_urls(query):
    """
    Naver 검색 API를 사용하여 이미지 URL 리스트를 반환합니다.
    """
    print(f"Fetching URLs for {query}...")
    urls = []
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    params = {
        "query": query,
        "display": 100, 
        "start": 1,
        "sort": "sim",
        "filter": "large"
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

    # 중복 방지를 위해 최종적으로 전체 폴더를 스캔하여 dataset.json을 다시 생성
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for name, query in TARGET_PEOPLE.items():
        person_dir = os.path.join(SAVE_DIR, query.replace(" ", "_"))
        
        # 이미 충분한 이미지가 있다면 스킵
        if os.path.exists(person_dir):
            existing_files = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if len(existing_files) >= IMAGES_PER_PERSON:
                print(f"Skipping {name}: already has {len(existing_files)} images.")
                continue
        else:
            os.makedirs(person_dir)
            
        urls = get_image_urls(query)
        
        saved_count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
        for url in urls:
            if saved_count >= IMAGES_PER_PERSON:
                break
                
            try:
                # 이미지 다운로드
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                
                # 파일 저장
                filename = f"{query.replace(' ', '_')}_{saved_count:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                img.save(filepath)
                
                saved_count += 1
                print(f"[{name}] Saved {saved_count}/{IMAGES_PER_PERSON}: {filename}")
                
            except Exception as e:
                continue

        print(f"Finished {name}: {saved_count} images total.")

    # dataset.json 갱신 (현재 폴더 상태를 기준으로 새로 작성)
    print("\nUpdating dataset.json based on actual files...")
    all_entries = []
    # TARGET_PEOPLE 순서대로 정렬하여 저장
    for name, query in TARGET_PEOPLE.items():
        person_dir = os.path.join(SAVE_DIR, query.replace(" ", "_"))
        if os.path.exists(person_dir):
            files = sorted([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
            for f in files:
                all_entries.append({
                    "image": os.path.join(person_dir, f),
                    "text_input": "이 인물은 누구입니까?",
                    "text_output": name
                })

    with open(DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset updated: {DATASET_JSON}")
    print(f"Total items in dataset: {len(all_entries)}")

if __name__ == "__main__":
    main()
