from transformers import pipeline

# Define NLP Pipelines
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")

# Function to simulate system and human messages
def chat_with_nlp(task_type, system_message, human_message, context=None):
    print("\n[System]:", system_message)

    if task_type == "summarization":
        summary = summarizer(human_message, max_length=50, min_length=10, do_sample=False)
        print("[AI]:", summary[0]['summary_text'])

    elif task_type == "qa":
        if not context:
            print("[AI]: Error! Context is required for Q&A task.")
            return
        answer = qa_pipeline(question=human_message, context=context)
        print("[AI]:", answer['answer'])

# Example 1: Summarization Task
chat_with_nlp(
    task_type="summarization",
    system_message="Provide a brief summary of the following passage:",
    human_message="Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding."
)

# Example 2: Question Answering Task
chat_with_nlp(
    task_type="qa",
    system_message="Extract the correct answer from the given passage.",
    human_message="What does AI aim to do?",
    context="Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding."
)
