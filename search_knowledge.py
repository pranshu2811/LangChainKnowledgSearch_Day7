import streamlit as st
from pinecone import Pinecone
from generate_embeddings import get_embedding
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def search_similar(query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        st.write("Failed to generate embedding for query.")
        return []
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    unique_results = []
    seen_texts = set()
    if "matches" in results:
        for match in results["matches"]:
            text = match['metadata']['text']
            score = match['score']
            if text not in seen_texts:
                unique_results.append((text, score))
                seen_texts.add(text)

    return unique_results
st.title("Search Your Custom Knowledge Base")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        results = search_similar(query)
        
        if results:
            st.subheader("Results:")
            for text, score in results:
                st.write(f"**{text}** (Score: {score:.4f})")
        else:
            st.write("No relevant content found.")
