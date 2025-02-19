from pinecone import Pinecone
from generate_embeddings import get_embedding
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def search_similar(query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    seen_texts = set()  
    if "matches" in results:
        for match in results["matches"]:
            text = match['metadata']['text']
            score = match['score']
            if text not in seen_texts:
                print(f"{text} {score:.4f}")
                seen_texts.add(text)  

if __name__ == "__main__":
    search_similar("AI and machine learning")
