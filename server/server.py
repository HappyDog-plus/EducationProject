from fastapi import FastAPI, File, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import time
from datetime import datetime
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
import logging
import os
import json
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests
import re
from server.xunfei import xunfei_recognize
from server.util import ModelWorker
from server.constants import patient_prompt, doctor_prompt, no_match

def current_time():
    t = datetime.now()
    return str(t.year)+"_"+str(t.month)+"_"+str(t.day)+"_"+str(t.hour)+"_"+str(t.minute)


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
                    filemode='a',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    encoding='utf-8'
                    )
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model backend is starting up...")
    # initialize chatbot (local_model: 0, openai_api: 1)
    app.state.model_type = 1
    app.state.modelworker = ModelWorker(model_type=1)
    # Exercise data
    ex_cat_path = data_dir / "ExerciseCategories.xlsx"
    ex_path = data_dir / "Exercises.xlsx"
    ex_cat_df = pd.read_excel(ex_cat_path, header=0)
    app.state.ex_keywords = ex_cat_df.iloc[:, 0].dropna().tolist()
    ex_df = pd.read_excel(ex_path)
    app.state.ex_questions = ex_df.iloc[:, 9].dropna().tolist()
    # Case report data
    cr_cat_path = data_dir / "CaseReportCategories.json"
    with open(cr_cat_path, "r") as f:
        app.state.cr_keywords = json.load(f)
    cr_path = data_dir / "CaseReports.json"
    with open(cr_path, "r") as f:
        app.state.case_reports = json.load(f)

    yield

    print("-"*50, "\nModel backend is closing ...\n", "-"*50)
    logger.info("Model backend is closing ...")


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
    audio_path = audio_save_path / file.filename
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # convert .wav to .mp3
    if file.filename.endswith(".wav"):
        # load .wav file
        audio = AudioSegment.from_wav(audio_path)
        # set sampling rate
        audio = audio.set_frame_rate(16000)
        audio_path0 = audio_save_path / (file.filename.split('.')[0] + ".mp3")
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


class Model_Data(BaseModel):
    user_id: str
    time_span: str
    mode_code: int
    input_text: str
    image: str


def match_kwds(text, kwd_list):
    llm = app.state.modelworker.model
    prompt = f"You are an expert in ophthalmology, and you need to complete the given tasks strictly in accordance with the format requirements.\
                        \nBased on the given list of categories, match the following text to similar categories\
                        \nCategories list: {', '.join(kwd_list)} \
                        \nText: \"{text}\" \
                        \nReturn format：{{\"keyword\": [matched categories list]}}"
    response = llm.invoke(prompt)
    if app.state.model_type == 1:
        response = response.content
    response = re.sub(r"```json|```", "", response)
    matched_kwds = json.loads(response)
    return matched_kwds["keyword"]


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
    chatbot = app.state.modelworker
    llm = app.state.modelworker.model

    if data.mode_code == 0:
        # GPT and local model have different style prompt.
        if app.state.model_type == 0:
            prompt = ""
            if data.image != "":
                prompt += ("<image>" + data.input_text + "<img>" + data.image + "</img>")
            else:
                prompt += data.input_text
            message = HumanMessage(content=prompt)
        else:
            if data.image != "":
                message = [
                            {"type": "text", "text": data.input_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{data.image}"},
                            },
                          ]
            else:
                message = data.input_text
        start = time.time()
        response = chatbot.invoke(usr_id=data.user_id, conv_type=0, msg=message)
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        # *** Image RAG not implemented
        return {"user_id": data.user_id, 
                "time_span": str(datetime.now()), 
                "mode_code": data.mode_code, 
                "output_text": response}

    elif data.mode_code == 1:
        user_id = data.user_id
        course_ids = ["26", "27", "38", "42"]
        question_ids = []
        url_excercise = "http://120.26.66.32:18080/api/v1/getExercise"
        start = time.time()
        for course_id in course_ids:
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
                            logger.info(f"Item in response does not contain 'question_id'.")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON for course_id {course_id}.")
            else:
                logger.error(f"Request for course_id {course_id} failed. Status code: {response.status_code}")
        question_history = sorted(set(question_ids))
        keyword_list = match_kwds(data.input_text, app.state.ex_keywords)
        question_list = app.state.ex_questions
        indices = [i + 1 for i, item in enumerate(question_list) if item in keyword_list]
        filtered_indices = set(indices) - set(question_history)
        string_indices = [str(index) for index in filtered_indices]
        result_indices = string_indices[:10]
        end = time.time()
        logger.info("Inference time: " + str(end - start))
        return {"user_id": data.user_id,
                "time_span": str(datetime.now()),
                "mode_code": data.mode_code,
                "question_ids": result_indices}

    elif data.mode_code == 2:
        keyword_list = match_kwds(data.input_text, app.state.cr_keywords)
        if len(keyword_list) == 0:
            return {
                        "user_id": data.user_id,
                        "time_span": str(datetime.now()),
                        "mode_code": data.mode_code,
                        "report_id": 0,
                        "concept": "",
                        "patient_text": no_match[datetime.now().second%len(no_match)]
                   }
        cr_idxs, cr_kws, cr_ctxs = [], [], []
        for item in app.state.case_reports:
            for kw in keyword_list:
                if kw == item["label"]:
                    cr_idxs.append(item["idx"])
                    cr_kws.append(kw)
                    cr_ctxs.append(item["context"])
        # randomly choose a case report
        index = datetime.now().second % len(cr_idxs)
        cr_id, cr_kw, cr_ctx = cr_idxs[index], cr_kws[index], cr_ctxs[index]
        prompt = patient_prompt.format(cr_ctx)
        response = llm.invoke(prompt).content
        chatbot.history_manager.save_message(usr=data.user_id, conv_type=1, msg_type="ai", content=response, time=str(datetime.now()))
        return {
                "user_id": data.user_id,
                "time_span": str(datetime.now()),
                "mode_code": data.mode_code,
                "report_id": cr_id+1,
                "concept": cr_kw,
                "patient_text": response
               }

    elif data.mode_code == 3:
        response = chatbot.invoke(usr_id=data.user_id, conv_type=1, msg=data.input_text)
        return {
            "user_id": data.user_id,
            "time_span": str(datetime.now()),
            "mode_code": data.mode_code,
            "patient_text": response
        }

    elif data.mode_code == 4:
        if "<d>" in data.input_text:
            response = chatbot.invoke(usr_id=data.user_id, conv_type=2, msg=data.input_text)
        else:
            q = re.findall('<q>(.*?)</q>', data.input_text)[0]
            c = re.findall('<c>(.*?)</c>', data.input_text)[0]
            prompt = doctor_prompt.format(q, c)
            response = llm.invoke(prompt).content
        return {
            "user_id": data.user_id,
            "time_span": str(datetime.now()),
            "mode_code": data.mode_code,
            "doctor_text": response
        }
    logger.info("Model Inference End")


class Course_Invoke_Data(BaseModel):
    user_id: str
    time_span: str
    question: str
    correct_ans: str
    user_ans: str


# Function to invoke OpenAI and get a semantic comparison
def evaluate_ans(question, correct_ans, user_ans):
    prompt = f'''
    Question: 
    {question}
    Correct Answer: 
    {correct_ans}
    Student Answer: 
    {user_ans}
    Task: Based on the correct answer provided, determine if the student answer is correct. If the student answer is correct, return 1; otherwise, return 0. Only provide the number 0 or 1 as the output.
    '''
    response = app.state.modelworker.model.invoke(prompt)
    if app.state.model_type == 1:
        response = response.content
    return response


# Function to invoke OpenAI and get an explanation
def get_explanation(question, correct_ans, user_ans):
    prompt = f'''
    Question: 
    {question}
    Correct answer: 
    {correct_ans}
    Student wrong answer: 
    {user_ans}
    Task: Evaluate the student answer based on the question and correct answer provided. Offer feedback that is objective, concise, logically clear.
    '''
    response = app.state.modelworker.model.invoke(prompt)
    if app.state.model_type == 0:
        response = response.content
    return response


@app.post("/course_invoke")
async def course_inference(data: Course_Invoke_Data):
    logger.info("Course Exercise Judgement Start")
    logger.info(
        "\nuser_id: " + data.user_id +
        "\ntime_span: " + data.time_span +
        "\nquestion: " + data.question +
        "\ncorrect_ans: " + data.correct_ans +
        "\nuser_ans: " + data.user_ans
    )
    res = int(evaluate_ans(data.question, data.correct_ans, data.user_ans))
    if not res:
        # Generate explanation if semantically incorrect
        explanation = get_explanation(data.question, data.correct_ans, data.user_ans)
    else:
        encouragements = [
            "Well done! Your answer is spot-on!",
            "Great job! Keep up the excellent work!",
            "You have a clear understanding and explained it perfectly!",
            "Fantastic! Your hard work is paying off!",
            "Exactly right! You're a smart cookie!",
            "Excellent answer! Keep going strong!",
            "Absolutely correct! You’ve truly mastered this!",
            "Outstanding work! Believe in yourself, you're amazing!"
        ]
        explanation = encouragements[datetime.now().second % 8]
    result = {
        "user_id": data.user_id,
        "time_span": str(datetime.now()),
        "res": res,
        "explanation": explanation
    }
    logger.info("Course Exercise Judgement End")
    return result


class Course_Gen_Data(BaseModel):
    template: str
    question: str


@app.post("/course_generate")
async def generate_data(data: Course_Gen_Data):
    # OMP error
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    logger.info("Course Generate Start")
    logger.info("question: " + data.question)
    llm0 = ChatOpenAI(
        model="gpt-4o-2024-05-13",
        temperature=0.2,
        max_tokens=4096,
        timeout=None
    )
    vectorstore_path = data_dir / "VectorStore" / "LectureContentVecs"
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
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


@app.post("/summary")
async def summary(data: summary_request_data):
    logger.info("Summary Start")
    # generate summary
    summary = "This is a summary."
    logger.info("Summary End")
    results = {"user_id": data.user_id, "time_span": str(datetime.now()), "summary": summary}
    return results