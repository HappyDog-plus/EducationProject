from fastapi import FastAPI, File, UploadFile
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llava_custom import Custom_LLaVA
from pydantic import BaseModel
# import uvicorn
import torch
# from langchain.schema import LLMResult
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import time
from datetime import datetime
import shutil
from xunfei import xunfei_recognize
from pathlib import Path
from contextlib import asynccontextmanager
from config import set_environment
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
import logging
import os
import json
import pandas as pd
import requests


def current_time():
    t = datetime.now()
    return str(t.year)+"_"+str(t.month)+"_"+str(t.day)+"_"+str(t.hour)+"_"+str(t.minute)


# create audio folder and log folder
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
audio_path = current_dir / "audio_saved"
log_path = current_dir / "log"
if not os.path.exists(audio_path):
    os.makedirs(audio_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)


# Create logger object
logging.basicConfig(
                    level=logging.INFO,
                    filename=Path("log") / (current_time() + ".log"),
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    encoding='utf-8'
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
    app.state.model_type = 1
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    if app.state.model_type == 0:
        model_path = r"/data/home/yangjiale/WorkSpace/EyeFM_Education/PretrainedModel/llava-v1.5-7b"
        model_name="llava-v1.5-7b"
        # Lora(set base model and adapter weights) 
        # base_path = "/workspace/LLaVA/PretrainedModel/llava-v1.5-7b"
        llm = Custom_LLaVA(model_path=model_path, 
                           model_base=None, 
                           model_name=model_name, 
                           load_4bit=False, 
                           load_8bit=True, 
                           device=device)
    else:
        set_environment()
        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2
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
    with_message_history.invoke([
                                    SystemMessage(content="A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions."), 
                                    AIMessage(content="Hello! How can I help you today?"),
                                    HumanMessage(content="Hi! Nice to meet you.")
                                ], 
                                config=config)
    # initialize global resources
    app.state.chatbot = with_message_history
    app.state.config_history = config
    # Exercise data
    ex_cat_path = current_dir.parent / "local_data" / "ExerciseCategories.xlsx"
    ex_path = current_dir.parent / "local_data" / "Exercises.xlsx"
    ex_cat_df = pd.read_excel(ex_cat_path, header=0)
    app.state.ex_keywords = ex_cat_df.iloc[:, 0].dropna().tolist()
    ex_df = pd.read_excel(ex_path)
    app.state.ex_questions = ex_df.iloc[:, 9].dropna().tolist()

    yield

    print("-"*50, "\nModel backend is closing ...\n", "-"*50)
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
    if not file.filename.endswith(('.wav', '.mp3')):
        return {"error": "File format not supported. Please upload a .wav or a .mp3 file."}
    logger.info("Audio Recognize Start")
    # save audio file
    audio_path = Path("audio_saved") / file.filename
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # convert .wav to .mp3
    if file.filename.endswith(".wav"):
        # load .wav file
        audio = AudioSegment.from_wav(audio_path)
        # set sampling rate
        audio = audio.set_frame_rate(16000)
        audio_path0 = Path("audio_saved") / (file.filename.split('.')[0] + ".mp3")
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
                "\nuser_id: " + data.user_id +  
                "\ntime_span: " + data.time_span + 
                "\nmode_code: " + str(data.mode_code) + 
                "\ninput_text: " + data.input_text + 
                "\nimage: " + data.image[-100:]
                )
    chatbot = app.state.chatbot
    config = app.state.config_history

    if data.mode_code == 0:
        prompt = ""
        # GPT and local model have different style prompt.
        if app.state.model_type == 0:
            if data.image != "":
                prompt += ("<image>" + data.input_text + "<img>" + data.image + "</img>")
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
                                                "image_url": {"url": f"data:image/jpeg;base64,{data.image}"},
                                            },
                                        ]
                                    )
            else:
                message = HumanMessage(content = data.input_text)
        start = time.time()
        response = chatbot.invoke(message, config=config)
        if app.state.model_type == 1:
            response = response.content
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        return {"user_id": data.user_id, 
                "time_span": str(datetime.now()), 
                "mode_code": data.mode_code, 
                "output_text": response}
    
    elif data.mode_code == 1:
        pass

    elif data.mode_code == 2:
        # class_path = "classfication.xlsx"
        # question_list_path = "result_1.xlsx"
        # df = pd.read_excel(class_path, header=0)
        llm = app.state.llm
        user_id = data.user_id  
        course_ids = ["26", "27", "38", "42"]
        question_ids = []
        url_excercise = "http://120.26.66.32:18080/api/v1/getExercise"
        for course_id in course_ids:
            # 构造请求的 JSON 数据
            param = {
                "user_id": user_id,
                "course_id": course_id
            }
            response = requests.get(url_excercise, params=param)
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    for item in response_data:
                        if 'question_id' in item:
                            question_ids.append(item['question_id'])
                        else:
                            print(f"Item in response does not contain 'question_id'.")
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for course_id {course_id}.")
            else:
                print(f"Request for course_id {course_id} failed. Status code: {response.status_code}")

            question_history = sorted(set(question_ids))

        keywords = app.state.ex_keywords
        prompt = f"You are an expert in ophthalmology, and you need to complete the given tasks strictly in accordance with the format requirements.\
                    \nBased on the given list of categories, match the following text to similar categories\
                    \nCategories list: {', '.join(keywords)} \
                    \nText: \"{data.input_text}\" \
                    \nReturn format：{{\"keyword\": [matching categories list]}}"
        message = HumanMessage(content = prompt)
        start = time.time()
        response = llm.invoke(message, config=config)
        if app.state.model_type == 1:
            response = response.content
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        classfication = json.loads(response)
        keyword_list = classfication["keyword"]
        # df = pd.read_excel(question_list_path)
        question_list = app.state.ex_questions
        indices = [i + 1 for i, item in enumerate(question_list) if item in keyword_list]

        filtered_indices = set(indices) - set(question_history)
        string_indices = [str(index) for index in filtered_indices]
        result_indices = string_indices[:10]
        return {"user_id": data.user_id, 
                "time_span": str(datetime.now()), 
                "mode_code": data.mode_code, 
                "question_ids": result_indices}
    
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
                    "\nuser_id: " + data.user_id + 
                    "\ntime_span: " + data.time_span + 
                    "\nquestion: " + data.question + 
                    "\ncorrect_ans: " + data.correct_ans + 
                    "\nuser_ans: " + data.user_ans
                )
    logger.info("Course Excercise Judgement End")
    return {
            "user_id": data.user_id,
            "time_span": str(datetime.now()),
            "res": datetime.now().second % 2
           }

    
