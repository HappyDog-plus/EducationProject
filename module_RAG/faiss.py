from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
import chardet

if __name__ == "__main__":
    vectorstore_path = Path("G:\Research\EyeFM_Education_Data\VectorStore\FAISS")
    vectorstore_path.mkdir(exist_ok=True)

    # convert docs to vectors, and persists vectorstore
    # docs_path = r"G:\Research\EyeFM_Education_Data\Data\TextBook"
    # doc_names = os.listdir(docs_path)
    # loaders = []
    # for doc_name in doc_names:
    #     doc_path = Path(docs_path) / doc_name
    #     with open(str(doc_path), "rb") as f:
    #         raw_data = f.read()
    #         result = chardet.detect(raw_data)
    #         encoding = result["encoding"]
    #         loaders.append(TextLoader(doc_path, encoding=encoding))
    # docs = []
    # for loader in loaders:
    #     docs.extend(loader.load())

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1080, chunk_overlap=20)
    # texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(texts, embeddings)
    # vectorstore.save_local(str(vectorstore_path))


    # Test
    vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke("Wills Eye Hospital?")
    print(docs)
    # start at 17:21 end at 17:34
