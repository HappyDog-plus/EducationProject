
from pydantic import BaseModel, Field
from typing import List, Dict
from pathlib import Path
import os, json
from langchain_openai import ChatOpenAI
import torch
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import trim_messages, BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from datetime import datetime
import re
from server.constants import system_messages
from server.llava_custom import Custom_LLaVA


# used path
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "local_data"


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    # In memory implementation of chat message history.
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        # Add a list of messages to the store
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class HistoryManager:

    def __init__(self):
        # every history file save no more than 1000 messages (not include system messages)
        self.__max_len = 400
        self.__current_file_path = Path(__file__).resolve().parent
        self.__chat_history_path = self.__current_file_path / "chat_history"
        if not os.path.exists(self.__chat_history_path):
            os.makedirs(self.__chat_history_path)
        all_items = os.listdir(self.__chat_history_path)
        self.__usrs = [item for item in all_items if os.path.isdir(os.path.join(self.__chat_history_path, item))]
        # 2 types of conversation types (0 normal, 1 case report patient, 2 case report doctor)
        self.__conv_types = [0, 1, 2]

    def __get_file(self, usr: str, conv_type: int) -> Path:
        usr_path = self.__chat_history_path / usr
        if not os.path.exists(usr_path):
            os.makedirs(usr_path)
            for t in self.__conv_types:
                self.__create_new_file(str(usr_path / f"{str(t)}_0.json"))
            return usr_path / (str(conv_type) + "_0.json")
        fs = [f for f in os.listdir(usr_path) if f.startswith(str(conv_type))]
        f_idxs = [int(f.split('.')[0].split('_')[-1]) for f in fs]
        return usr_path / (str(conv_type) + "_" + str(max(f_idxs)) + ".json")

    @staticmethod
    def __create_new_file(f_path: str) -> None:
        with open(f_path, "w") as f:
            json.dump([], f)

    def save_message(self, usr: str, conv_type: int, msg_type: str, content: str, time: str) -> None:
        f_p = self.__get_file(usr, conv_type)
        with open(f_p, "r") as f:
            msgs = json.load(f)
        # if file length larger than the max_len, create a new file
        if len(msgs) >= self.__max_len:
            f_n = int(f_p.stem.split('_')[-1]) + 1
            f_p = self.__chat_history_path / usr / (str(conv_type) + "_" + str(f_n) + ".json")
            self.__create_new_file(str(f_p))
            msgs = []
        item = {
                    "time": time,
                    "msg_type": msg_type,
                    "content": content
               }
        msgs.append(item)
        with open(f_p, "w") as f:
            json.dump(msgs, f, indent=4)

    def load_messages(self, usr: str, conv_type: int) -> List[BaseMessage]:
        # if accidently the file is small, ignore content in the previous file
        f_p = self.__get_file(usr, conv_type)
        with open(f_p, "r") as f:
            msgs = json.load(f)
        res = []
        for msg in msgs:
            if msg["msg_type"] == "ai":
                res.append(AIMessage(content=msg["content"]))
            elif msg["msg_type"] == "human":
                res.append(HumanMessage(content=msg["content"]))
        return res

    def load_history(self) -> Dict:
        history = {}
        for usr in self.__usrs:
            for t in self.__conv_types:
                history[(usr, t)] = InMemoryHistory()
                history[(usr, t)].add_messages([SystemMessage(content=system_messages[t])])
                usr_ai_msgs = self.load_messages(usr, t)
                history[(usr, t)].add_messages(usr_ai_msgs)
        return history

    def make_sure_usr(self, usr: str) -> bool:
        if not usr in self.__usrs:
            usr_path = self.__chat_history_path / usr
            os.makedirs(usr_path)
            for t in self.__conv_types:
                self.__create_new_file(str(usr_path / f"{str(t)}_0.json"))
            self.__usrs.append(usr)
            return False
        else:
            return True


class ModelWorker:
    def __init__(self, model_type: int):
        # 0 openai api, 1 local model
        self.model_type = model_type
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model_type == 0:
            model_path = data_dir / "ModelWeights" / "llava-v1.5-7b"
            model_name = "llava-v1.5-7b"
            # Lora(set base model and adapter weights)
            # base_path = "/workspace/LLaVA/PretrainedModel/llava-v1.5-7b"
            self.model = Custom_LLaVA(model_path=model_path,
                                       model_base=None,
                                       model_name=model_name,
                                       load_4bit=False,
                                       load_8bit=True,
                                       device=device)
        else:
            self.model = ChatOpenAI(
                                        model="gpt-4o",
                                        temperature=0,
                                        max_tokens=None,
                                        timeout=None,
                                        max_retries=2
                                    )
            # caching layer for chat models (reducing the number of API calls)
            set_llm_cache(InMemoryCache())
        self.history_manager = HistoryManager()
        self.user_chat_history = self.history_manager.load_history()
        self.chat_bot = RunnableWithMessageHistory(
                                                        runnable=self.model,
                                                        get_session_history=self.__get_user_history,
                                                        history_factory_config=[
                                                            ConfigurableFieldSpec(
                                                                id="user_id",
                                                                annotation=str,
                                                                name="User ID",
                                                                description="Unique identifier for the user.",
                                                                default="",
                                                                is_shared=True,
                                                            ),
                                                            ConfigurableFieldSpec(
                                                                id="conv_type",
                                                                annotation=int,
                                                                name="Conversation ID",
                                                                description="Unique identifier for the conversation.",
                                                                default="",
                                                                is_shared=True,
                                                            ),
                                                        ],
                                                     )

    def __get_user_history(self, user_id: str, conv_type: int) -> InMemoryHistory:
        self.user_chat_history[(user_id, conv_type)].messages = trim_messages(
                                                                                self.user_chat_history[(user_id, conv_type)].messages,
                                                                                strategy="last",
                                                                                token_counter=len,
                                                                                max_tokens=5,
                                                                                # start_on="human",
                                                                                include_system=True
                                                                             ) 
        return self.user_chat_history[(user_id, conv_type)]

    def __format_sys_msg(self, usr: str, conv_type: int, msg: str):
        # parse related information
        c = re.findall('<c>(.*?)</c>', msg)
        q = re.findall('<q>(.*?)</q>', msg)
        d = re.findall('<d>(.*?)</d>', msg)
        # bug: each time we need to format the background information
        self.user_chat_history[(usr, conv_type)].messages[0].content = system_messages[conv_type]
        if len(q) == 0:
            # patient stage
            self.user_chat_history[(usr, conv_type)].messages[0].content = self.user_chat_history[(usr, conv_type)].messages[0].content.format(c[0])
        else:
            self.user_chat_history[(usr, conv_type)].messages[0].content = self.user_chat_history[(usr, conv_type)].messages[0].content.format(c[0], q[0], d[0])

    def __invoke(self, usr_id: str, conv_type: int, human_msg: HumanMessage) -> str:
        config = {"configurable": {"user_id": usr_id, "conv_type": conv_type}}
        response = self.chat_bot.invoke(human_msg, config)
        if self.model_type == 1:
            response = response.content
        return response

    def clear_history(self, usr_id: str, conv_type: int):
        self.user_chat_history[(usr_id, conv_type)].clear()
        # bug: After clear msg queue, a system msg should be appended
        self.user_chat_history[(usr_id, conv_type)].add_messages([SystemMessage(content=system_messages[conv_type])])
        
    def invoke(self, usr_id: str, conv_type: int, msg: str) -> str:
        # if the usr never use system, create usr folder and 0 files, reload history
        if not self.history_manager.make_sure_usr(usr_id):
            self.user_chat_history = self.history_manager.load_history()
        if conv_type == 0:
            human_msg = HumanMessage(content=msg)
        else:
            # no need to load history
            self.__format_sys_msg(usr=usr_id, conv_type=conv_type, msg=msg)
            human_msg = HumanMessage(content=(re.findall('<s>(.*?)</s>', msg)[0]))
        response = self.__invoke(usr_id, conv_type, human_msg)
        # bug: Should write to disk after each invoke. Otherwise, quit program the last message will be saved twice.
        # If response successfully, at least 2 messages in the msg queue
        if response:
            last_request = self.user_chat_history[(usr_id, conv_type)].messages[-2]
            self.history_manager.save_message(usr_id, conv_type, "human", last_request.content, str(datetime.now()))
            last_response = self.user_chat_history[(usr_id, conv_type)].messages[-1]
            self.history_manager.save_message(usr_id, conv_type, "ai", last_response.content, str(datetime.now()))
        return response

if __name__ == '__main__':
    from config import set_environment
    set_environment()
    pass