from google.cloud import firestore

# Firestore Setup
PROJECT_ID = "langchain-9644f"  # Replace with your Firebase project ID
SESSION_ID = "user_session_new"  # This could be a username or unique identifier
COLLECTION_NAME = "chat_history"  # Firestore collection name

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
class FirestoreChatMessageHistory:
    def __init__(self, session_id, collection, client):
        self.session_id = session_id
        self.collection = collection
        self.client = client
        self.messages = self._load_messages()

    def _load_messages(self):
        doc_ref = self.client.collection(self.collection).document(self.session_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("messages", [])
        return []

    def save_message(self, sender, message):
        doc_ref = self.client.collection(self.collection).document(self.session_id)
        new_message = {"sender": sender, "message": message, "timestamp": firestore.SERVER_TIMESTAMP}
        doc_ref.set({"messages": firestore.ArrayUnion([new_message])}, merge=True)
        self.messages.append(new_message)

# Initialize Chat History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)

print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Example: Chat Loop
print("Start chatting with the AI. Type 'exit' to quit.")
while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.save_message(sender="user", message=human_input)
    print("Message saved to Firestore.")
