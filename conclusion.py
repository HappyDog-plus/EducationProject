import openai
import pandas as pd
import json

import requests

# 请求参数定义
params_38 = {
    "user_id": "1",
    "course_id": "38",
    # 添加其他需要的参数
}
params_42 = {
    "user_id": "1",
    "course_id": "42",
    # 添加其他需要的参数
}
params_27 = {
    "user_id": "1",
    "course_id": "27",
    # 添加其他需要的参数
}
params_26 = {
    "user_id": "1",
    "course_id": "26",
    # 添加其他需要的参数
}
params_QA = {
    "user_id": "1",
    # 添加其他需要的参数
}

# 获取笔记内容
# 定义 API 端点
url_note = "http://120.26.66.32:18080/api/v1/getNotes"

# 发送 GET 请求
response_note26 = requests.get(url_note, params=params_26)
response_note27 = requests.get(url_note, params=params_27)
response_note38 = requests.get(url_note, params=params_38)
response_note42 = requests.get(url_note, params=params_42)
# 检查响应状态
if response_note26.status_code == 200:
    # 解析 JSON 数据
    note_26 = response_note26.json()
    # print(note_26)
else:
    print(f"Error: {response_note26.status_code}")
if response_note27.status_code == 200:
    # 解析 JSON 数据
    note_27 = response_note27.json()
    # print(note_27)
else:
    print(f"Error: {response_note27.status_code}")

if response_note38.status_code == 200:
    # 解析 JSON 数据
    note_38 = response_note38.json()
    # print(note_38)
else:
    print(f"Error: {response_note38.status_code}")
if response_note42.status_code == 200:
    # 解析 JSON 数据
    note_42 = response_note42.json()
    # print(note_42)
else:
    print(f"Error: {response_note42.status_code}")

# 查询案例分析

# 定义 API 端点
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
# 定义 API 端点
url_excercise = "http://120.26.66.32:18080/api/v1/getExercise"

# 发送 GET 请求
response_ti26 = requests.get(url_excercise, params=params_26)
response_ti27 = requests.get(url_excercise, params=params_27)
response_ti38 = requests.get(url_excercise, params=params_38)
response_ti42 = requests.get(url_excercise, params=params_42)
# 检查响应状态
if response_ti26.status_code == 200:
    # 解析 JSON 数据
    question_26 = response_ti26.json()
    # print(question_26)
else:
    print(f"Error: {response_ti26.status_code}")
# 检查响应状态
if response_ti27.status_code == 200:
    # 解析 JSON 数据
    question_27 = response_ti27.json()
    # print(question_27)
else:
    print(f"Error: {response_ti27.status_code}")
if response_ti38.status_code == 200:
    # 解析 JSON 数据
    question_38 = response_ti38.json()
    # print(question_38)
else:
    print(f"Error: {response_ti38.status_code}")
if response_ti42.status_code == 200:
    # 解析 JSON 数据
    question_42 = response_ti42.json()
    # print(question_42)
else:
    print(f"Error: {response_ti42.status_code}")

# 查询问答记录
# 定义 API 端点
url_QA = "http://120.26.66.32:18080/api/v1/getChatHistory"

# 定义请求参数（根据接口文档修改具体参数）


# 发送 GET 请求
response = requests.get(url_QA, params=params_QA)

# 检查响应状态
if response.status_code == 200:
    # 解析 JSON 数据
    Q_A = response.json()
else:
    print(f"Error: {response.status_code}")

openai.api_key = ""


# 上课对话总结的函数
# /api/v1/getChatHistory 查询问答记录 参数：user_id
def chat_conclude(conversations):
    prompt = f"According to the conversations '{conversations}' provided by me, the part of text are the questions I raised in the ophthalmology class and the answers given by the teacher. Please help me summarize the questions and key words for future reference, the word number is between 100-200 words,no need to output other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用GPT-4.0 Mini模型
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 学习报告
# 1.	上课弹窗提问涉及知识点（回答不对的/答不全）
# 2.	练习错误题目总结 （题干加知识点）/api/v1/getExcercise查询用户已答题目  参数user_id，course_id
def excercise_conclude(description):
    prompt = f"According to the conclusion '{description}' provided by me, if the parameters of answer and user_ans are inconsistent, concise summary of their topic knowledge and explanation and then output, no need to ouput correct answer and user answer and other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用GPT-4.0 Mini模型
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 3.	学生笔记 （直接复制学生所有笔记整理一下格式）/api/v1/getNotes 参数：user_id,course_id
def notes_conclude(note):
    prompt = f"According to the notebook '{note}' provided by me, the content behind the uword necrosis keyword are my class notes from the ophthalmology course. Please help me organize the knowledge points in outline form and output it, no need to output other words"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用GPT-4.0 Mini模型
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


def generate():

    print("Learning report")
    print("Q&A summary")
    print(chat_conclude(Q_A))
    print("\ncellular adaptation")
    print("Error summary")
    print(excercise_conclude(question_26))
    print("Note summary:")
    print(notes_conclude(note_26))
    print("\nnecrosis")
    print("Error summary")
    print(excercise_conclude(question_27))
    print("Note summary:")
    print(notes_conclude(note_27))
    print("\n Red Eye")
    print("Error summary")
    print(excercise_conclude(question_38))
    print("Note summary:")
    print(notes_conclude(note_38))
    print("\n 255")
    print("Error summary")
    print(excercise_conclude(question_42))
    print("Note summary:")
    print(notes_conclude(note_42))


generate()
