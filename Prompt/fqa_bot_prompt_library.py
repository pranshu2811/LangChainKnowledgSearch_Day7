import os
import requests
from pinecone import Pinecone
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
import numpy as np

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist in the Pinecone environment '{PINECONE_ENVIRONMENT}'.")

index = pinecone_client.Index(PINECONE_INDEX_NAME)

def query_pinecone(vector, top_k=5):
    try:
        return index.query(vector=vector, top_k=top_k)
    except Exception as e:
        raise RuntimeError(f"Error querying Pinecone: {e}")

def generate_ai_response(prompt):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "prompt": prompt,
            "max_tokens": 200  
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()  
    except Exception as e:
        raise RuntimeError(f"Error generating AI response from Gemini API: {e}")

def main():
    try:
    
        input_vector = np.random.rand(768).tolist()
        print("Querying Pinecone...")
        pinecone_results = query_pinecone(vector=input_vector, top_k=5)
        print("Pinecone Results:", pinecone_results)

        prompt = "What are the best practices for resolving application crashes?"
        print("Generating AI response...")
        ai_response = generate_ai_response(prompt)
        print("AI Response:", ai_response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
