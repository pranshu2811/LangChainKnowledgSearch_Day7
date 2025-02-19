from prompt_library import PromptLibrary
prompt_lib = PromptLibrary()
print(prompt_lib.get_prompt("embedding", "generate", text="What is Pinecone?"))
print(prompt_lib.get_prompt("similarity_search", "debug", query_text="Find similar articles about AI", top_k=5))
print(prompt_lib.get_prompt("system", "setup"))
