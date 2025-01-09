from src.medrag import MedRAG

if __name__ == "__main__":
    # Test MedRAG
    
    # question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
    # options = {
    #     "A": "paralysis of the facial muscles.",
    #     "B": "paralysis of the facial muscles and loss of taste.",
    #     "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    #     "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
    # }

    # medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
    # answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
    # print(f"Final answer in json with rationale: {answer}")

    # Initialize customized textbooks
    # rev = Retriever(retriever_name="bm25", corpus_name="customized_textbooks")
    # print(rev.get_relevant_documents(question="what is red eye?"))

    # Test Retriever System
    # rs = RetrievalSystem(retriever_name="RRF-4", corpus_name="Textbooks")
    # print(rs.retrieve(question="How to differentiate between CRVO and hypertensive retinoparthy?", k=16))

    # Test MedRAG
    medrag = MedRAG(llm_name="OpenAI/gpt-4o-2024-11-20", rag=True, retriever_name="RRF-4", follow_up=True, corpus_name="Textbooks")
    print(medrag.answer(question="the nasolacrimal canal extends into what part of the nose?", options=None, n_rounds=2, n_queries=2, save_path="./messages"))