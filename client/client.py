import requests
import base64
from PIL import Image
import io
from typing import Any, Dict, Iterator, List, Optional, Sequence
from datetime import datetime
from pathlib import Path


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
    

    # url = "https://4okpi1941999.vicp.fun/course_invoke"
#     url = "http://127.0.0.1:8001/model"

    # # Test model
    # Test model conversation
    # img_path = Path("image") / "test_img.jpg"
    # img_base64 = img_encode(img_path)
    # data0 = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(0), "input_text": "Describe this image.", "image": img_base64}
    # response = requests.post(url, json=data0)
    # print(response.json())

    # Test searching exercises based on student input.
#     data = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(2), "input_text": "I want to do some exercise about Glaucoma", "image": ""}
#     response = requests.post(url, json=data)
#     print(response.json())


    # Test XunFei Recognize API
    # file_path = Path("audio") / "test_wav.wav"
    # with open(file_path, 'rb') as f:
    #     response = requests.post(url, files={"file": f})
    # print(response.json())


    # Test in-class exercise judge model.
    # data1 = {
    #             "user_id": "12345678",
    #             "time_span": "2024-09-20 16:43:20.0000",
    #             "question": "What's the answer of 1 + 1?",
    #             "correct_ans": "The answer is 2.",
    #             "user_ans": "I don't know."
    #         }
    # response = requests.post(url, json=data1)
    # print(response.json())


    # Test 7: Course video window QA
    url = " http://127.0.0.1:8000/course_invoke"
    data = {
                "user_id": "123",
                "time_span": "2024-09-20 16:43:20.0000",
                "question": "What’s the answer of 1 + 1?",
                "correct_ans": "The answer is 2.",
                "user_ans": "Maybe the answer is 2, I guess."
           }
    response = requests.post(url, json=data)
    print(response.json())
