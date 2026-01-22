import requests
import json
import base64
import sys

# vLLM Server Configuration
API_URL = "http://localhost:18001/v1/chat/completions"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "/dataset/cep/llm-training/person-test/merged_model"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_image_inference(image_path):
    base64_image = encode_image(image_path)
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": "이 인물은 누구입니까?"}
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }

    print(f"Sending request to {API_URL} with image: {image_path}...")
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        content = result['choices'][0]['message']['content']
        print(f"\n[Model Answer]: {content}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Details: {e.response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        print("Example: python test_api.py data/test.jpg")
    else:
        test_image_inference(sys.argv[1])
