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
    

    url = "https://4okpi1941999.vicp.fun/course_invoke"
    # url = "http://0.0.0.0:8000/model"

    # # Test model
    # img_path = r"image/test_img.jpg"
    # img_base64 = img_encode(img_path)
    # data0 = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(0), "input_text": "Describe this image: ", "image": img_base64}
    # response = requests.post(url, json=data0)
    # if response.status_code == 200:
    #     result = response.json()
    #     print(result)
    # else:
    #     print("Failed: ", response.status_code)


    # Test XunFei Recognize API
    # file_path = r"audio/test_wav.wav"
    # with open(file_path, 'rb') as f:
    #     response = requests.post(url, files={"file": f})
    # print(response.json())


    # Test in-class exercise judge model.
    data1 = {
                "user_id": "12345678",
                "time_span": "2024-09-20 16:43:20.0000",
                "question": "What’s the answer of 1 + 1?",
                "correct_ans": "The answer is 2.",
                "user_ans": "I don’t know."
            }
    response = requests.post(url, json=data1)
    print(response.json())