import os
import csv
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def open_csv(path):
    with open(file_path, "r", encoding="utf-8") as f:
        datareader = csv.reader(f, delimiter=";")
        for row in tqdm(datareader, desc="Splitting text"):
            if len(row) > 3:  # Ensure row[3] exists
                post_text = row[3]
                chunks = text_splitter.split_text(post_text)
                documents.extend(chunks)
# Paths and configurations
s = "sd"
vector_store_path = "./chroma_persist_" + s

if s == "mh":
    file_path = r"C:\RAG\new_post_mentalhealth_241119.csv"
if s == "li":
    file_path = r"C:\RAG\new_post_Lithium_241119.csv"
if s == "sd":
    file_path = r"C:\RAG\new_post_SEXONDRUGS_241121.csv"
if s == "rad":
    file_path = r"xxx"
file_name, file_extension = os.path.splitext(file_path)

# 1. Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Lightweight embedding model
answering_model = Ollama(model="incept5/llama3.1-claude",
                         base_url="http://127.0.0.1:11434")  # High-quality answering model
# 2. Adjust chunking for fewer splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
print(f"Loading and splitting data from: {file_path}")

# 3. Load and split documents
documents = []
if(file_extension == ".csv"):
    open_csv(file_path)

print(f"Total chunks created: {len(documents)}")
doc_objects = [Document(page_content=chunk) for chunk in documents]

# 4. Create or load the vector store
if os.path.exists(vector_store_path):
    print("Loading existing vector store...")
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model
    )
else:
    print("Creating a new vector store...")
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model
    )

    # Batch embeddings for faster processing
    batch_size = 32
    batches = [doc_objects[i:i + batch_size] for i in range(0, len(doc_objects), batch_size)]
    print("Adding embeddings in batches...")
    for batch in tqdm(batches, desc="Embedding batches"):
        texts = [doc.page_content for doc in batch]
        vector_store.add_texts(texts=texts)

print("Vector store setup complete.")

# 5. Configure retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 100})

# 6. Create retrieval chain with answering model
retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are a mental health researcher analyzing posts from an online community about mental health concerns. 
Your role is to extract insights based on the posts provided and deliver responses grounded only in those posts, avoiding assumptions or generic information. 
Respond in a professional and concise manner, focusing on trends and themes in the data. You're helping practitioners identify trends among patients and people taking risks.

Context: {context}

Question: {input}
Response:
"""
)
combine_docs_chain = create_stuff_documents_chain(
    llm=answering_model,
    prompt=retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query = "Based on the context, answer the following questions : What are the main molecules people use for having sex on drugs and which percentage of the posts do they represent ? What percentage of the post mention at least one molecule used for chemsex ? What are the users mainly inquiring about ?"
print("Running query...")
response = retrieval_chain.invoke({"input": query})
print(f"Response: {response['answer']}")
