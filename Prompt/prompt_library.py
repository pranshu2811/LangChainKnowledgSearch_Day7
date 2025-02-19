from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

class PromptLibrary:
    def __init__(self):
        self.prompts = {
            "embedding": {
                "generate": "Generate an embedding vector for the following text: '{text}'",
                "analyze": "Analyze the embedding vector and explain its meaning: {vector}",
            },
            "similarity_search": {
                "query": "Find items similar to the following input using a vector database: '{query_text}'",
                "debug": "Explain the similarity search process for input query: '{query_text}' with top_k={top_k}",
            },
            "system": {
                "setup": f"Configure Pinecone with the API key '{PINECONE_API_KEY}' and index '{PINECONE_INDEX_NAME}'.",
                "test_connection": f"Test the connection to Pinecone with the API key '{PINECONE_API_KEY}'.",
            },
            "debugging": {
                "error_handling": "Given the error '{error_message}', suggest a solution for a Pinecone-related application.",
                "validate_index": f"Validate the setup of Pinecone index '{PINECONE_INDEX_NAME}' and ensure all embeddings are correctly stored.",
            }
        }

    def get_prompt(self, category: str, key: str, **kwargs) -> str:
        """
        Retrieve a formatted prompt.

        Args:
            category (str): The category of the prompt.
            key (str): The key within the category.
            **kwargs: Parameters to format the prompt.

        Returns:
            str: The formatted prompt.
        """
        try:
            prompt_template = self.prompts[category][key]
            return prompt_template.format(**kwargs)
        except KeyError:
            raise ValueError(f"Invalid category '{category}' or key '{key}'.")
        except KeyError as e:
            raise ValueError(f"Missing required parameter for formatting: {e}")
