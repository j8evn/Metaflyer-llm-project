import os
import requests
from PIL import Image
from io import BytesIO
import json

NAVER_CLIENT_ID = "UvhrrH5qgzxFlnYFmkTf"
NAVER_CLIENT_SECRET = "ICOBFihYe7"
API_ENDPOINT = "https://openapi.naver.com/v1/search/image"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEOPLE_LIST_JSON = os.path.join(BASE_DIR, "config/people.json")
SAVE_DIR = os.path.join(BASE_DIR, "data/test") 

TEST_IMAGES_COUNT = 5 

def load_target_people():
    if os.path.exists(PEOPLE_LIST_JSON):
        with open(PEOPLE_LIST_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

TARGET_PEOPLE = load_target_people()

def get_image_urls(query):
    """
    Naver 검색 API를 사용하여 이미지 URL 리스트 반환
    """
    print(f"Fetching Test URLs for {query}...")
    urls = []
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    params = {
        "query": query,
        "display": 20, 
        "start": 41,   
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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for name, query in TARGET_PEOPLE.items():
        person_dir = os.path.join(SAVE_DIR, query.replace(" ", "_"))
        
        # 이미 충분히 수집되었다면 스킵
        if os.path.exists(person_dir):
            existing_files = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if len(existing_files) >= TEST_IMAGES_COUNT:
                print(f"Skipping Test {name}: already has {len(existing_files)} images.")
                continue
        else:
            os.makedirs(person_dir)
            
        urls = get_image_urls(query)
        
        saved_count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
        for url in urls:
            if saved_count >= TEST_IMAGES_COUNT:
                break
                
            try:
                # 이미지 다운로드
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                
                # 파일 저장
                filename = f"test_{query.replace(' ', '_')}_{saved_count:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                img.save(filepath)
                
                saved_count += 1
                print(f"[{name}] Saved Test {saved_count}/{TEST_IMAGES_COUNT}: {filename}")
                
            except Exception as e:
                continue

        print(f"Finished {name}: {saved_count} test images total.")

if __name__ == "__main__":
    main()
