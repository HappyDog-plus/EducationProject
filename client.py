import requests
import base64
from PIL import Image
import io
from typing import Any, Dict, Iterator, List, Optional, Sequence



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
    # url = "http://127.0.0.1:21002"
    url = "https://4okpi1941999.vicp.fun"
    img_path = r"image\54.jpg"
    img_base64 = img_encode(img_path)
    # data = { "input_text": "My name is Peter. Nice to see you.", "image": ""}
    # break   +   delete_history
    data = { "input_text": "I have some questions to ask you? Are you available now?", "image": "", "status": 0}
    # data = { "input_text": "My coffee bar open. Can you write a slogan for me?", "image": ""}
    # data = { "input_text": "Can you describe the image for me.", "image": img_base64}
    # data = { "input_text": "OK. That's a very nice slogan. Thank you very much.", "image": ""}
    # data = { "input_text": "Can you give me another interesting one?", "image": ""}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("(Return response)\n", result['message'])
    else:
        print("Failed: ", response.status_code)