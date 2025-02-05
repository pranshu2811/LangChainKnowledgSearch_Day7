import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".html"):
            loader = BSHTMLLoader(file_path)
        else:
            continue
        docs.extend(loader.load())

    return docs

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"] if "embedding" in response else None
    except Exception:
        return None

def store_embeddings(text_chunks):
    vectors = []
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk.page_content)
        if embedding:
            vector_id = f"doc_{i}"
            vectors.append({"id": vector_id, "values": embedding, "metadata": {"text": chunk.page_content}})
    
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Stored {len(vectors)} document chunks in Pinecone.")
        print("\n First 3 stored document chunks:")
        for v in vectors[:3]:
            print(v["metadata"]["text"][:100]) 


if __name__ == "__main__":
    docs = load_documents("documents")  
    chunks = split_text(docs) 
    store_embeddings(chunks)  

