from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
# import asyncio
import warnings
warnings.filterwarnings("ignore")
import base64
from PIL import Image
import io
import re
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, BaseMessage, Generation
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
import torch



class Custom_LLaVA(LLM):
    model_name:        str = None
    temperature:       float = 0.7
    top_p:             float = 0.1
    max_new_tokens:    int = 512
    model:             Any = None
    tokenizer:         Any = None
    image_processor:   Any = None
    callback_manager:  Any = None

    def __init__(self, model_path, model_name, load_8bit, load_4bit, device, model_base=None, **kwargs):
        super().__init__()
        self.model_name = str(model_name)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                                                                                        model_path=model_path, 
                                                                                        model_base=model_base, 
                                                                                        model_name=model_name, 
                                                                                        load_8bit=load_8bit, 
                                                                                        load_4bit=load_4bit, 
                                                                                        device=device
                                                                                    )

    # Sequence[BaseMessage] |
    def invoke(
                self,
                input: ChatPromptValue,
                history: Dict[str, Any],
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs
              ) -> Union[LLMResult, str]:
        if isinstance(input, ChatPromptValue):
            input = input.messages
        # Delete the image base64 part of the input prompt
        human_prompt = input[-1].content
        human_prompt, imgs = self.preprocess_prompt(human_prompt)
        # set the input window size
        conv_window_size = 2
        history_convs = input[:-1] if len(input) < conv_window_size else input[-conv_window_size : -1]
        prompt = self.combine_messages_list(history_convs) + "HUMAN: " + human_prompt + "\nAI:"
        # stop word
        stopwords = ["HUMAN", "human", "Human", "SYSTEM", "System", "system", "USER", "user", "User"]
        output = self._call(prompt=prompt, images=imgs)
        output = self.postprocess_response(response=output, stopwords=stopwords)
        # Maybe run with no history
        if history['configurable']:    
            history['configurable']['message_history'].add_user_message(human_prompt)
            history['configurable']['message_history'].add_ai_message(output)
        # return LLMResult(generations=[[Generation(text=output)]])
        return output

    def combine_messages_list(self, messages: List[BaseMessage]) -> str:
        full_text = ""
        for msg in messages:
            full_text += (str(msg.type).upper() + ": " + msg.content + "\n")
        # remove the <image> label in previous conversations
        full_text = re.sub("<image>", "", full_text)
        return full_text

    def postprocess_response(self, response: str, stopwords: List[str]) -> str:
        response1 = re.sub("ai:", "", response, re.IGNORECASE)
        # prevent generating HUMAN message
        min_index = len(response1)
        for stopword in stopwords:
            index = response1.find(stopword)
            if index != -1 and index < min_index:
                min_index = index
        if min_index != len(response1):
            response2 = response1[:min_index]
        else:
            response2 = response1 
        return response2.strip()

    def preprocess_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        pattern = re.compile(r'<img>.*?</img>', re.DOTALL)
        imgs_with_label = pattern.findall(prompt)
        prompt = pattern.sub("", prompt)
        imgs = []
        for img in imgs_with_label:
            img = re.sub("^<img>", "", img)
            img = re.sub("</img>$", "", img)
            imgs.append(img)
        return (prompt, imgs)

    def _call(
                self,
                prompt: str,
                images: Optional[List[str]] = None,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
             ) -> str:
        temperature, top_p, max_new_tokens = self.temperature, self.top_p, self.max_new_tokens
        is_multimodal = 'llava' in self.model_name.lower()

        if images is not None and len(images) > 0 and is_multimodal:
            if len(images) > 0:
                images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in images]
                images = process_images(images, self.image_processor, self.model.config)
                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)
                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            else:
                images = None
                image_sizes = None
            image_args = {"images": images, "image_sizes": image_sizes}
        else:
            images = None
            image_args = {}

        temperature = float(temperature)
        top_p = float(top_p)
        max_new_tokens = min(int(max_new_tokens), 1024)
        do_sample = True if temperature > 0.001 else False
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        
        output = self.model.generate(
                        inputs=input_ids,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        **image_args
                      )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def _stream(
                    self,
                    prompt: str,
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> Iterator[GenerationChunk]:
        output = self._call(prompt)

        chunk_size = 4
        for i in range(0, len(output), chunk_size):
            yield GenerationChunk(text=output[i:i + chunk_size])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return { }

    @property
    def _llm_type(self) -> str:
        return "LLaVA(MultiModal Large Language Model)"


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_base64_str = img_base64.decode("utf-8")
    return img_base64_str



def test():
    # image_path = r"E:\SJTU_Medical\LLaVA\serve_images\2024-08-19\5a603cbcf864db01aa2d5841e01ed4f2.jpg"
    # image = Image.open(image_path)
    # image_base64 = image_to_base64(image)
    # images = [image_base64]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "E:\\SJTU_Medical\\LLaVA\\pretrained_model\\llava-v1.5-7b"
    # model_name="llava-v1.5-7b"
    # llm = Custom_LLaVA(model_path=model_path, model_base=None, model_name=model_name, load_4bit=False, load_8bit=True, device=device)
    

    # Version 1 (Can Run like the original code. Plus, local model can be connected to the Langchain.)
    # img_prompt = ChatPromptTemplate.from_messages(
    #                                             [
    #                                                 ("system", "A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions."),
    #                                                 ("human", "<image>{prompt}\nAI:<image_placeholder>{images}")
    #                                             ]
    #                                          )
    # chain = img_prompt | llm

    # print(chain.invoke({"prompt": "Write a story as long as possible to describe this image.", "images": image_base64}))
    # for chunk in chain.stream({"prompt": "Write a story as long as possible to describe this image.", "images": image_base64}):
    #     print(chunk, end="")
    

    # Version 2 (Can use it to record previous chat using RunnableWithMessageHistory from Langchain)
    # store = {}
    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     if session_id not in store:
    #         store[session_id] = InMemoryChatMessageHistory()
    #     return store[session_id]
    # with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    # config = {"configurable": {"session_id": "idx1"}}
    # response = with_message_history.invoke([
    #                                          SystemMessage(content="A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions."), 
    #                                          AIMessage(content="Hello! How can I help you today?"),
    #                                          HumanMessage(content="Hi! I'm Peter.")
    #                                        ], 
    #                                          config=config)
    # response = with_message_history.invoke(HumanMessage(content="What's my name?"), config=config)
    # for generation in response.generations:
    #     for gen in generation:
    #         print(gen.text)
    # response = with_message_history.invoke(HumanMessage(content="Tell me a funny story about a sheep."), config=config)
    # for generation in response.generations:
    #     for gen in generation:
    #         print(gen.text)

    # version 3 (Combine with other toolkit, api)

    # Test prompt preprocess
    # text =  "<div><img>Image 1 content</img><p>Some text</p><img>Image 2 content</img><img>Image 3 content</img></div>"
    # prompt, imgs = preprocess_prompt(text)
    # print(prompt)
    # print(imgs)
    pass



if __name__ == "__main__":
    test()
    pass