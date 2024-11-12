from fastapi import FastAPI, File, UploadFile
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from server.llava_custom import Custom_LLaVA
from pydantic import BaseModel

# import uvicorn
import torch
import openai

# from langchain.schema import LLMResult
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import time
from datetime import datetime
import shutil
from server.xunfei import xunfei_recognize
from pathlib import Path
from contextlib import asynccontextmanager
from server.config import set_environment
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
import logging
import os
import json
import requests
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def current_time():
    t = datetime.now()
    return (
        str(t.year)
        + "_"
        + str(t.month)
        + "_"
        + str(t.day)
        + "_"
        + str(t.hour)
        + "_"
        + str(t.minute)
    )


# create audio folder and log folder
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
data_dir = current_dir.parent / "local_data"
audio_save_path = current_dir / "audio_saved"
log_path = current_dir / "log"
if not os.path.exists(audio_save_path):
    os.makedirs(audio_save_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)


# Create logger object
logging.basicConfig(
    level=logging.INFO,
    filename=log_path / (current_time() + ".log"),
    filemode="a",
    datefmt="%Y/%m/%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class Model_Data(BaseModel):
    user_id: str
    time_span: str
    mode_code: int
    input_text: str
    image: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model backend is starting up...")
    # initialize chatbot (local_model: 0, openai_api: 1)
    app.state.model_type = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if app.state.model_type == 1:
        model_path = data_dir / "ModelWeights" / "llava-v1.5-7b"
        model_name = "llava-v1.5-7b"
        # Lora(set base model and adapter weights)
        # base_path = "/workspace/LLaVA/PretrainedModel/llava-v1.5-7b"
        llm = Custom_LLaVA(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_4bit=False,
            load_8bit=True,
            device=device,
        )
    else:
        set_environment()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    # caching layer for chat models (reducing the number of API calls)
    set_llm_cache(InMemoryCache())
    # record history chat
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    config = {"configurable": {"session_id": "idx1"}}
    # initialize chat
    with_message_history.invoke(
        [
            SystemMessage(
                content="A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions."
            ),
            AIMessage(content="Hello! How can I help you today?"),
            HumanMessage(content="Hi! Nice to meet you."),
        ],
        config=config,
    )
    # initialize global resources
    app.state.chatbot = with_message_history
    app.state.config_history = config
    app.state.llm = llm
    # Exercise data
    ex_cat_path = data_dir / "ExerciseCategories.xlsx"
    ex_path = data_dir / "Exercises.xlsx"
    ex_cat_df = pd.read_excel(ex_cat_path, header=0)
    app.state.ex_keywords = ex_cat_df.iloc[:, 0].dropna().tolist()
    ex_df = pd.read_excel(ex_path)
    app.state.ex_questions = ex_df.iloc[:, 9].dropna().tolist()

    yield

    print("-" * 50, "\nModel backend is closing ...\n", "-" * 50)
    logger.info("Model backend is closing ...")
    del llm
    del with_message_history
    del device


app = FastAPI(lifespan=lifespan)


def delete_file(file_path):
    try:
        file = Path(file_path)
        if file.is_file():
            file.unlink()
            logger.info(f"File {file_path} is deleted successfully.")
        else:
            logger.warning(f"File {file_path} does not exist.")
    except Exception as e:
        logger.error(f"File deleting error: {e}")


# convert .wav audio to texts
@app.post("/recognize")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3")):
        return {
            "error": "File format not supported. Please upload a .wav or a .mp3 file."
        }
    logger.info("Audio Recognize Start")
    # save audio file
    audio_path = audio_save_path / file.filename
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # convert .wav to .mp3
    if file.filename.endswith(".wav"):
        # load .wav file
        audio = AudioSegment.from_wav(audio_path)
        # set sampling rate
        audio = audio.set_frame_rate(16000)
        audio_path0 = audio_save_path / (file.filename.split(".")[0] + ".mp3")
        # export .mp3 audio file
        audio.export(audio_path0, format="mp3")
        # delete .wav file
        delete_file(audio_path)
        audio_path = audio_path0
    # recognize texts
    result, error_code = xunfei_recognize(audio_path)
    # delete audio file, release space
    delete_file(audio_path)
    logger.info("\nText:" + result + "\nError_Code:" + str(error_code))
    logger.info("Audio Recognize End")
    return {"output_text": result, "error_code": error_code}


@app.post("/model")
async def model_inference(data: Model_Data):

    logger.info("Model Inference Start")
    logger.info(
        "\nuser_id: "
        + data.user_id
        + "\ntime_span: "
        + data.time_span
        + "\nmode_code: "
        + str(data.mode_code)
        + "\ninput_text: "
        + data.input_text
        + "\nimage: "
        + data.image[-100:]
    )
    chatbot = app.state.chatbot
    config = app.state.config_history
    llm = app.state.llm

    if data.mode_code == 0:
        prompt = ""
        # GPT and local model have different style prompt.
        if app.state.model_type == 1:
            if data.image != "":
                prompt += "<image>" + data.input_text + "<img>" + data.image + "</img>"
            else:
                prompt += data.input_text
            message = HumanMessage(content=prompt)
        else:
            if data.image != "":
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": data.input_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{data.image}"
                            },
                        },
                    ]
                )
            else:
                message = HumanMessage(content=data.input_text)
        start = time.time()
        response = chatbot.invoke(message, config=config)
        if app.state.model_type == 0:
            response = response.content
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        return {
            "user_id": data.user_id,
            "time_span": str(datetime.now()),
            "mode_code": data.mode_code,
            "output_text": response,
        }

    elif data.mode_code == 1:
        pass

    elif data.mode_code == 2:
        keywords = app.state.ex_keywords
        prompt = f"You are an expert in ophthalmology, and you need to complete the given tasks strictly in accordance with the format requirements.\
                    \nBased on the given list of categories, match the following text to similar categories\
                    \nCategories list: {', '.join(keywords)} \
                    \nText: \"{data.input_text}\" \
                    \nReturn format：{{\"keyword\": [matching categories list]}}"
        # message = HumanMessage(content = prompt)
        start = time.time()
        response = llm.invoke(prompt)
        if app.state.model_type == 0:
            response = response.content
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        classification = json.loads(response)
        # Debug
        print(classification)
        keyword_list = classification["keyword"]
        question_list = app.state.ex_questions
        indices = [
            i + 1 for i, item in enumerate(question_list) if item in keyword_list
        ]
        result_indices = indices[:10]
        string_indices = [str(index) for index in result_indices]
        return {
            "user_id": data.user_id,
            "time_span": str(datetime.now()),
            "mode_code": data.mode_code,
            "question_ids": string_indices,
        }

    elif data.mode_code == 3:
        pass

    else:
        pass
    logger.info("Model Inference End")


class Course_Invoke_Data(BaseModel):
    user_id: str
    time_span: str
    question: str
    correct_ans: str
    user_ans: str


@app.post("/course_invoke")
async def model_inference(data: Course_Invoke_Data):
    logger.info("Course Excercise Judgement Start")
    logger.info(
        "\nuser_id: "
        + data.user_id
        + "\ntime_span: "
        + data.time_span
        + "\nquestion: "
        + data.question
        + "\ncorrect_ans: "
        + data.correct_ans
        + "\nuser_ans: "
        + data.user_ans
    )
    logger.info("Course Excercise Judgement End")
    return {
        "user_id": data.user_id,
        "time_span": str(datetime.now()),
        "res": datetime.now().second % 2,
    }


class Course_Gen_Data(BaseModel):
    template: str
    question: str


@app.post("/course_generate")
async def generate_data(data: Course_Gen_Data):
    # OMP error
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logger.info("Course Generate Start")
    logger.info("question: " + data.question)
    llm0 = ChatOpenAI(
        model="gpt-4o-2024-05-13", temperature=0.2, max_tokens=4096, timeout=None
    )
    vectorstore_path = data_dir / "VectorStore" / "LectureContentVecs"
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        str(vectorstore_path), embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )
    # prompt_template = """
    #                     Generate a lecture script based on the following context:

    #                     Context:
    #                     {context}

    #                     Question:
    #                     {question}

    #                     Please use clear and professional language, ensuring that the script is logically coherent, accurate, and suitable for student learning. Include an introduction, main content, and a conclusion.
    #                   """
    prompt_template = data.template
    prompt = ChatPromptTemplate.from_messages([("user", prompt_template)])

    def format_docs(docs):
        # Debug
        # print(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    # Debug
    # def print_prompt(prompt):
    #     print("Intermediate prompt results", prompt)
    #     return prompt

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        # Debug
        # | RunnableLambda(print_prompt)
        | llm0
        | StrOutputParser()
    )

    docs = retriever.invoke(data.question)
    docs_list = [doc.page_content for doc in docs]
    response = rag_chain.invoke(str(data.question))
    logger.info("Course Generate End")
    results = {"response": response, "documents": docs_list}
    return results


class summary_request_data(BaseModel):
    user_id: str
    course_id: str
    time_span: str


# 获取笔记内容
url_note = "http://120.26.66.32:18080/api/v1/getNotes"


# 笔记总结
def notes_conclude(note):
    prompt = f"According to the notebook '{note}' provided by me, the content behind the uword necrosis keyword are my class notes from the ophthalmology course. Please help me organize the knowledge points in outline form and output it, no need to output other words"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 查询案例分析
url_anli = "http://120.26.66.32:18080/api/v1/getExampleAnalysis"
# 查询用户已答题目
url_excercise = "http://120.26.66.32:18080/api/v1/getExercise"


# 题目总结
def excercise_conclude(description):
    prompt = f"According to the conclusion '{description}' provided by me, if the parameters of answer and user_ans are inconsistent, concise summary of their topic knowledge and explanation and then output, no need to ouput correct answer and user answer and other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


# 查询问答记录
url_QA = "http://120.26.66.32:18080/api/v1/getChatHistory"


# 问答总结
def chat_conclude(conversations):
    prompt = f"According to the conversations '{conversations}' provided by me, the part of text are the questions I raised in the ophthalmology class and the answers given by the teacher. Please help me summarize the questions and key words for future reference, the word number is between 100-200 words,no need to output other words"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 使用GPT-4.0 Mini模型
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"].strip()


@app.post("/summary")
async def summary(data: summary_request_data):
    logger.info("Summary Start")
    params = {
        "user_id": data.user_id,
        "course_id": data.course_id,
    }
    response_note = requests.get(url_note, params=params)
    note = response_note.json()
    response_ti = requests.get(url_excercise, params=params)
    question = response_ti.json()
    response_qa = requests.get(url_QA, params=params)
    qa = response_qa.json()
    logger.info("Summary End")
    results = {
        "user_id": data.user_id,
        "time_span": str(datetime.now()),
        "Q&A summary": qa,
        "Error summary": question,
        "Note summary": note,
    }
    return results
