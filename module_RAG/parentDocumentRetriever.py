from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
import pickle

from server.config import set_environment
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path


if __name__ == "__main__":
    docs_path = r""
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    vectorstore_path = current_dir / "vectors"
    # Step 1: Load documents
    doc_names = os.listdir(docs_path)
    loaders = [TextLoader(Path(docs_path) / doc_name) for doc_name in doc_names]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    # save docs in .pkl
    docs_folder = current_dir / "docs"
    docs_folder.mkdir(exist_ok=True)
    docstore_path = current_dir / "docs" / "docstore.pkl"

    # Step 2: Define text splitters for parent and child documents
    # chunk_size is the character size. English words average length are 5. So 216 tokens(words) is about 1080
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2160)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1080)

    # Step 3: Initialize Chroma with persistence
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(
                            collection_name="split_parents",
                            embedding_function=embedding,
                            persist_directory=str(vectorstore_path)  # Enable persistence
                        )

    # Step 4: Check if persisant docstore exists
    if os.path.exists(docstore_path):
        # load docstore
        with open(docstore_path, "rb") as f:
            store = pickle.load(f)
    else:
        # initialize docstore
        store = InMemoryStore()

    # Step 5: Create or load the retriever
    retriever = ParentDocumentRetriever(
                                            vectorstore=vectorstore,
                                            docstore=store,
                                            child_splitter=child_splitter,
                                            parent_splitter=parent_splitter,
                                        )

    # Step 5: Add documents, save to disk only the first time
    # retriever.add_documents(docs)
    # with open(docstore_path, "wb") as f:
    #     pickle.dump(store, f)

    # Test
    # sub_docs = vectorstore.similarity_search("Cotton wool spots")
    # print(sub_docs[0].page_content)
    retrieved_docs = retriever.invoke("Cotton wool spots")
    print(retrieved_docs[0].page_content)