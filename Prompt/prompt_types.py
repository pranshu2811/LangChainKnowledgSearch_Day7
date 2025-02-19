import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
missing_vars = []
if not GEMINI_API_KEY:
    missing_vars.append("GEMINI_API_KEY")
if not PINECONE_API_KEY:
    missing_vars.append("PINECONE_API_KEY")
if not PINECONE_ENVIRONMENT:
    missing_vars.append("PINECONE_ENVIRONMENT")
if not PINECONE_INDEX_NAME:
    missing_vars.append("PINECONE_INDEX_NAME")

if missing_vars:
    raise ValueError(
        f"The following environment variables are missing in the .env file: {', '.join(missing_vars)}"
    )
def zero_shot_prompt(prompt):
    """
    Handles Zero-Shot prompting where no examples are provided.
    """
    print("\n[Zero-Shot Prompt]")
    print(prompt)
    return f"Simulated response for Zero-Shot: '{prompt}'"


def few_shot_prompt(prompt, examples):
    """
    Handles Few-Shot prompting where a few examples are provided.
    """
    print("\n[Few-Shot Prompt]")
    few_shot_input = "\n".join([f"- {inp} -> {out}" for inp, out in examples])
    final_prompt = f"{few_shot_input}\n{prompt} ->"
    print(final_prompt)
    return f"Simulated response for Few-Shot: '{final_prompt}'"


def chain_of_thought_prompt(prompt):
    """
    Handles Chain-of-Thought prompting where reasoning is explicit.
    """
    print("\n[Chain-of-Thought Prompt]")
    print(prompt)
    # Call to Gemini API or print simulated reasoning process
    return f"Simulated step-by-step response for Chain-of-Thought: '{prompt}'"
if __name__ == "__main__":
    zero_shot = zero_shot_prompt("Translate the following sentence into French: 'How are you?'")
    print(zero_shot)
    few_shot = few_shot_prompt(
        "Translate 'How are you?' into French.",
        examples=[
            ("Hello", "Bonjour"),
            ("Good morning", "Bon matin")
        ]
    )
    print(few_shot)
    chain_of_thought = chain_of_thought_prompt(
        """
        If I have 3 apples and I buy 2 more, how many do I have?
        Let's think step by step:
        1. Start with 3 apples.
        2. Add 2 apples to the 3.
        3. Total apples = ?
        """
    )
    print(chain_of_thought)
