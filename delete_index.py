# check_indexes.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print("Current indexes:")
indexes = pc.list_indexes()
for index in indexes:
    print(f"- {index['name']} (dimension: {index['dimension']}, metric: {index['metric']})")

# Delete specific indexes if needed
indexes_to_delete = ["medical-chatbot", "medical-chatbot-hf"]
for index_name in indexes_to_delete:
    if index_name in [idx['name'] for idx in indexes]:
        print(f"\nDeleting index: {index_name}")
        pc.delete_index(index_name)
        print(f"Index {index_name} deleted successfully")