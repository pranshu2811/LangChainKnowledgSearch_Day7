import os

print("\n Running setup_pinecone.py...")
os.system("python setup_pinecone.py")

print("\n Running generate_embeddings.py...")
os.system("python generate_embeddings.py")

print("\n Running store_embeddings.py...")
os.system("python store_embeddings.py")

print("\n Running search.py...")
os.system("python search.py")

print("\n All scripts executed successfully!")
