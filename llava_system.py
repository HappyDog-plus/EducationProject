import torch
from llava_custom import Custom_LLaVA
from config import set_environment

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import LLMResult



def get_response_text(response: LLMResult) -> str:
    full_text = ""
    for generation in response.generations:
        for gen in generation:
            full_text += gen.text
    return full_text



if __name__ == "__main__":
    # initialize chatbot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # add API KEY
    set_environment()
    # load model
    model_path = "G:\\Research\\EyeFM_Education\\LLaVA\\pretrained_model\\llava-v1.5-7b"
    model_name="llava-v1.5-7b"
    llm = Custom_LLaVA(model_path=model_path, model_base=None, model_name=model_name, load_4bit=False, load_8bit=True, device=device)
    
    # 1. Load, chunk and index the contents of the blog to create a retriever.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    # print(docs[0].metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise."
                        "\n\n"
                        "{context}"
                    )

    prompt = ChatPromptTemplate.from_messages(
                                                [
                                                    ("system", system_prompt),
                                                    ("human", "{input}"),
                                                ]
                                              )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": "What is Task Decomposition?"})
    print(response["answer"])