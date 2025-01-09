from fastapi import FastAPI
from src.medrag import MedRAG
from pydantic import BaseModel
import json
import re

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.rag = MedRAG(
                            llm_name="OpenAI/gpt-4o-2024-11-20", 
                            rag=True, 
                            retriever_name="RRF-2", 
                            follow_up=True, 
                            corpus_name="Textbooks"
                          )
    app.state.n_rounds = 2
    app.state.n_queries = 1

@app.on_event("shutdown")
async def shutdown_event():
    pass

class RAG_Data(BaseModel):
    medical_question: str

@app.post("/rag_answer")
async def rag_answer(data: RAG_Data):
    msg_save_path = "./messages"
    result, msgs = app.state.rag.answer(
                                    question=data.medical_question, 
                                    options=None, 
                                    n_rounds=app.state.n_rounds, 
                                    n_queries=app.state.n_queries, 
                                    save_path=msg_save_path
                                  )
    # print(result)
    # print(msgs[-3]["content"])
    matches = re.findall(r"## Analysis(.+?)## Answer", msgs[-3]["content"], re.DOTALL)
    analysis = matches[0] if len(matches) else ""
    matches = re.findall(r"## Answer(.*)", msgs[-3]["content"], re.DOTALL)
    result = matches[0] if len(matches) else ""
    return {
            "result": result, 
            "analysis": analysis
           }


