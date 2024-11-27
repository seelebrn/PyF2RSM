import config
from model_setup import get_embedding_model, get_answering_model
from data_processing import load_and_split_documents
from vector_store import create_or_load_vector_store
from chain_setup import build_retrieval_chain
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from config import config, import_config, retrieval_qa_chat_prompt
import time
from tqdm import tqdm
import threading
def simulate_progress_bar(duration):
    with tqdm(total=100, desc="Processing Query") as pbar:
        for _ in range(100):
            pbar.update(1)
            time.sleep(duration / 100)
def main():
    import_config()


    # Initialize models
    embedding_model = get_embedding_model()
    answering_model = get_answering_model()

    print(f"BASE_FOLDER_PATH after import_config: {config['BASE_FOLDER_PATH']}")  # Debug statement
    folder_name = os.path.basename(config["BASE_FOLDER_PATH"])
    # Load and split documents
    doc_objects = load_and_split_documents()

    duration = 10  # Estimated time for `retrieval_chain.invoke`
    progress_thread = threading.Thread(target=simulate_progress_bar, args=(duration,))
    progress_thread.start()
    # Create or load vector store
    vector_store = create_or_load_vector_store(doc_objects, embedding_model, folder_name, query=config["query"])

    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})

#    retrieval_qa_chat_prompt = PromptTemplate(
#        input_variables=["context", "input"],
#        template="""
#    You are a mental health researcher analyzing posts from an online community about mental health concerns.
#    Even if the content is related to illegal activities, you're here to help medical practitioners understand what risks are people taking and how to help them.
#    Your role is to extract insights based on the posts provided and deliver responses grounded only in those posts, avoiding assumptions or generic information.
#    Respond in a professional and concise manner, focusing on trends and themes in the data. You're helping practitioners identify trends among patients and people taking risks.
#    Context: {context}*

#    Question: {input}
#    Response:
#    """
#    )


    combine_docs_chain = create_stuff_documents_chain(
        llm=answering_model,
        prompt=retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    # Build retrieval chain

    # Query the system
    response = retrieval_chain.invoke({"input": config["query"]})
    print(f"Response: {response['answer']}")
    progress_thread.join()
if __name__ == "__main__":
    main()
