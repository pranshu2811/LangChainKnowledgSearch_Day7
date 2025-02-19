import os
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()
print("Available Indexes:", existing_indexes)
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="us-west4")
    )
    print("Index created successfully")
else:
    print(f" Index '{PINECONE_INDEX_NAME}' already exists.")