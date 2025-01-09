import requests
import base64
from PIL import Image
import io
from typing import Any, Dict, Iterator, List, Optional, Sequence
from datetime import datetime
from pathlib import Path
import json


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
    # url_root = "http://43.129.224.174:6000"
    url_root = "http://127.0.0.1:8001"
    # url_root = ""

    # Test: model
    # url = url_root + "/model"
    # Test 1: model conversation
    # img_path = Path("image") / "test_img.jpg"
    # img_base64 = img_encode(img_path)
    # data = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(0), "input_text": "hello", "image": ""}

    # Test 2: searching exercises based on student input.
    # data = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(1), "input_text": "I want to do some exercises about Glaucoma", "image": ""}

    # Test 3: case report request
    # data = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(2), "input_text": "I want to do some case report exercise about Presbyopia", "image": ""}
    # response = requests.post(url, json=data)
    # print(response.json())


    # Test 4: case report patient
    # context = "A 48-year-old woman with myopia complains of progressive deterioration in distance and near vision in both eyes for the past 2 years. She can improve her vision by sliding her glasses down her nose. Her past medical history is significant for diabetes for 10 years, for which she takes glyburide. She reports blood sugar levels usually between 120 and 140 mg/dL and a recent hemoglobin A1c of 6.8%."
    # data = { "user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(3), "input_text": f"<s>Have a good day!</s><c>{context}</c>", "image": ""}
    # response = requests.post(url, json=data)
    # print(response.json())

    # Test 5: case report doctor
    # input_text = f'''
    #              <q>What is the problem?</q>
    #              <c>{context}</c>
    #              '''

    # input_text = f'''
    #                  <q>What is the problem?</q>
    #                  <c>{context}</c>
    #                  <s>I don't know the answer.</s>
    #                  <d>Presbyopia</d>
    #              '''
    # data = {"user_id": "userInfo1", "time_span": str(datetime.now()), "mode_code": int(4), "input_text": input_text, "image": ""}
    # response = requests.post(url, json=data)
    # print(response.json())



    # Test 3: XunFei Recognize API
    # url = url_root + "/recognize"
    # file_path = Path("audio") / "test_wav.wav"
    # print(str(file_path))
    # with open(file_path, 'rb') as f:
    #     response = requests.post(url, files={"file": f})
    # print(response.json())


    # Test 4: in-class exercise judge model.
    # url = url_root + "/course_invoke"
    # data = {
    #     "user_id": "123",
    #     "time_span": "2024-09-20 16:43:20.0000",
    #     "question": "What’s the answer of 1 + 1?",
    #     "correct_ans": "The answer is 2.",
    #     "user_ans": "Maybe the answer is 2, I guess."
    # }
    # response = requests.post(url, json=data)
    # print(response.json())


    # Test 5: generating course
    # url = url_root + "/course_generate"
    # data = {
    #             "template": """
    #                             Generate a lecture script based on the following context:

    #                             Context:
    #                             {context}

    #                             question:
    #                             {question}

    #                             Please use clear and professional language, ensuring that the script is logically coherent, accurate, and suitable for student learning. Include an introduction, main content, and a conclusion.
    #                         """,

    #             "question": "What is eye?"

    #        }
    # response = requests.post(url, json=data)

    # t = datetime.now()
    # output_path = str(t.year)+"_"+str(t.month)+"_"+str(t.day)+"_"+str(t.hour)+"_"+str(t.minute)+str(t.second)+ ".json"
    # with open(output_path, 'w') as json_file:
    #     json.dump(response.json(), json_file, indent=4)


    # Test 6: Summary
    # url = url_root + "/summary"
    # data = {
    #             "user_id": "12345678",
    #             "time_span": "2024-09-20 16:43:20.0000",
    #             "course_id": "1",
    #         }
    # response = requests.post(url, json=data)
    # print(response.json())

    # Test 8: Medrag
    url = url_root + "/rag_answer"
    data = {
            "medical_question": "the nasolacrimal canal extends into what part of the nose?"
           }
    response = requests.post(url, json=data)

    pass
