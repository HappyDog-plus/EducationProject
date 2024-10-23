import requests
import base64
from PIL import Image
import io
from typing import Any, Dict, Iterator, List, Optional, Sequence
from datetime import datetime


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_base64_str = img_base64.decode("utf-8")
    return img_base64_str



def img_encode(img_path: str) -> str:
    image = Image.open(img_path)
    return image_to_base64(image)



if __name__ == "__main__":
    

    # url = "https://4okpi1941999.vicp.fun/model"
    url = "http://localhost:8000/recognize"

    # # Test AI 

    # img_path = r".\image\test_img.jpg"
    # img_base64 = img_encode(img_path)
    # data0 = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(0), "input_text": "Can you remember my name?", "image": ""}
    # response = requests.post(url, json=data0)
    # if response.status_code == 200:
    #     result = response.json()
    #     print(result)
    #     # print("(Return response)\n", result['message'])
    # else:
    #     print("Failed: ", response.status_code)

    # data1 = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(0), "input_text": "Do you know my name ?", "image": ""}
    # response = requests.post(url, json=data1)
    # if response.status_code == 200:
    #     result = response.json()
    #     print(result)
    #     # print("(Return response)\n", result['message'])
    # else:
    #     print("Failed: ", response.status_code)

    # Test XunFei Recognize API
    file_path = r"audio\test_wav.wav"
    with open(file_path, 'rb') as f:
        response = requests.post(url, files={"file": f})
    print(response.json())