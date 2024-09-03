from fastapi import FastAPI
from fastapi import BackgroundTasks
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llava_custom import Custom_LLaVA
from pydantic import BaseModel
import uvicorn
import torch
from langchain.schema import LLMResult


from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import time


class RequestData(BaseModel):
    input_text: str
    image: str 



app = FastAPI()



@app.post("/")
async def generate_answer(data: RequestData):
    prompt = ""
    if data.image != "":
        prompt += ("<image>" + data.input_text + "<img>" + data.image + "</img>")
    else:
        prompt += data.input_text
    start = time.time()
    response = with_message_history.invoke([HumanMessage(content=prompt)], config=config)
    end = time.time()
    print("Inference time: ", end - start)
    # Extract response text from LLMResult object
    full_text = get_response_text(response)
    return {"message": full_text}



def get_response_text(response: LLMResult) -> str:
    full_text = ""
    for generation in response.generations:
        for gen in generation:
            full_text += gen.text
    return full_text



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



if __name__ == "__main__":
    # initialize chatbot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "H:\\Research\\EyeFM_Education\\LLaVA\\pretrained_model\\llava-v1.5-7b"
    model_name="llava-v1.5-7b"
    llm = Custom_LLaVA(model_path=model_path, model_base=None, model_name=model_name, load_4bit=False, load_8bit=True, device=device)
    
    # caching layer for chat models (reducing the number of API calls)
    set_llm_cache(InMemoryCache())
    
    # record history chat
    store = {}
    with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    config = {"configurable": {"session_id": "idx1"}}
    response = with_message_history.invoke([
                                             SystemMessage(content="A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions."), 
                                             AIMessage(content="Hello! How can I help you today?"),
                                             HumanMessage(content="Hi! Nice to meet you.")
                                           ], 
                                            config=config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
