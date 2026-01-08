import os
import requests
from PIL import Image
from io import BytesIO

NAVER_CLIENT_ID = "UvhrrH5qgzxFlnYFmkTf"
NAVER_CLIENT_SECRET = "ICOBFihYe7"
API_ENDPOINT = "https://openapi.naver.com/v1/search/image"
SAVE_DIR = "data/test_images" # 테스트용 이미지 저장 폴더

# 테스트할 인물 리스트 (학습 데이터와 동일한 인물)
TARGET_PEOPLE = {
    "박나래": "박나래",
    "아이유": "아이유",
    "유재석": "유재석",
    "일론 머스크": "Elon Musk",
    "제니": "블랙핑크 제니",
    "정국": "방탄소년단 정국"
}
IMAGES_PER_PERSON = 5 # 인물당 5장씩만 수집

def get_image_urls(query, count):
    """
    Naver 검색 API를 사용하여 이미지 URL 리스트를 반환합니다.
    학습 데이터와 겹치지 않게 하기 위해 start 파라미터를 조정합니다.
    """
    print(f"Fetching Test URLs for {query}...")
    urls = []
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    # 학습 데이터가 100개였으므로, 101번째부터 검색하여 겹침 방지
    params = {
        "query": query,
        "display": count,
        "start": 101, 
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
                
                # 파일 저장 (test_ 접두사 추가)
                filename = f"test_{query.replace(' ', '_')}_{i:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                img.save(filepath)
                
                count += 1
                print(f"Saved {filepath}")
                
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        print(f"Finished {name}: {count} test images saved.")

if __name__ == "__main__":
    main()
