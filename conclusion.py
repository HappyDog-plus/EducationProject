import openai
import pandas as pd
import json
import requests

params_38 = {
    "user_id": "1",
    "course_id": "38",
    
}
params_42 = {
    "user_id": "1",
    "course_id": "42",
}
params_27 = {
    "user_id": "1",
    "course_id": "27",
}
params_26 = {
    "user_id": "1",
    "course_id": "26",
}
params_QA = {
    "user_id": "1",
}

# 获取笔记内容
url_note = "http://120.26.66.32:18080/api/v1/getNotes"

response_note26 = requests.get(url_note, params=params_26)
response_note27 = requests.get(url_note, params=params_27)
response_note38 = requests.get(url_note, params=params_38)
response_note42 = requests.get(url_note, params=params_42)
note_26 = response_note26.json()
note_27 = response_note27.json()
note_38 = response_note38.json()
note_42 = response_note42.json()
# 查询案例分析
url_anli = "http://120.26.66.32:18080/api/v1/getExampleAnalysis"

# 定义请求参数（根据接口文档修改具体参数）
params = {
    "concept": "1",  # 案例类型
    # 添加其他需要的参数
}

# 发送 GET 请求
response = requests.get(url_anli, params=params)

# 检查响应状态
if response.status_code == 200:
    # 解析 JSON 数据
    data = response.json()
    # print(data)
else:
    print(f"Error: {response.status_code}")

# 查询用户已答题目
url_excercise = "http://120.26.66.32:18080/api/v1/getExercise"
response_ti26 = requests.get(url_excercise, params=params_26)
response_ti27 = requests.get(url_excercise, params=params_27)
response_ti38 = requests.get(url_excercise, params=params_38)
response_ti42 = requests.get(url_excercise, params=params_42)

question_26 = response_ti26.json()
question_27 = response_ti27.json()
question_38 = response_ti38.json()
question_42 = response_ti42.json()

# 查询问答记录
url_QA = "http://120.26.66.32:18080/api/v1/getChatHistory"
response_qa26 = requests.get(url_QA, params=params_26)
response_qa27 = requests.get(url_QA, params=params_27)
response_qa38 = requests.get(url_QA, params=params_38)
response_qa42 = requests.get(url_QA, params=params_42)

qa_26 = response_qa26.json()
qa_27 = response_qa27.json()
qa_38 = response_qa38.json()
qa_42 = response_qa42.json()

openai.api_key = ""


# 对话
def chat_conclude(conversations):
    prompt = f"According to the conversations '{conversations}' provided by me, the part of text are the questions I raised in the ophthalmology class and the answers given by the teacher. Please help me summarize the questions and key words for future reference, the word number is between 100-200 words,no need to output other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用GPT-4.0 Mini模型
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 题目总结
def excercise_conclude(description):
    prompt = f"According to the conclusion '{description}' provided by me, if the parameters of answer and user_ans are inconsistent, concise summary of their topic knowledge and explanation and then output, no need to ouput correct answer and user answer and other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 笔记总结
def notes_conclude(note):
    prompt = f"According to the notebook '{note}' provided by me, the content behind the uword necrosis keyword are my class notes from the ophthalmology course. Please help me organize the knowledge points in outline form and output it, no need to output other words"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


 class summary_request_data(BaseModel): 
     user_id: str 
     course_id: str 
     time_span: str 
  
  
 @app.post("/summary") 
 async def summary(data: summary_request_data): 
     logger.info("Summary Start") 
     qa_26 = chat_conclude(qa_26)
     qa_27 = chat_conclude(qa_27)
     qa_38 = chat_conclude(qa_38)
     qa_42 = chat_conclude(qa_42)
     question_26 = excercise_conclude(question_26)
     question_27 = excercise_conclude(question_27)
     question_38 = excercise_conclude(question_38)
     question_42 = excercise_conclude(question_42)
     note_26 = notes_conclude(note_26)
     note_27 = notes_conclude(note_27)
     note_38 = notes_conclude(note_38)
     note_42 = notes_conclude(note_42)
     logger.info("Summary End") 
     results = {"user_id": data.user_id, "time_span": str(datetime.now()), 
                "class_name":"cellular adaptation","Q&A summary": qa_26,"Error summary":question_26,"Note summary":note_26,
                "class_name":"necrosis","Q&A summary": qa_27,"Error summary":question_27,"Note summary":note_27,
                "class_name":"Red Eye","Q&A summary": qa_38,"Error summary":question_38,"Note summary":note_38,
                "class_name":"255","Q&A summary": qa_42,"Error summary":question_42,"Note summary":note_42,} 
     return results 
