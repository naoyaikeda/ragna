from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os

load_dotenv()

client = ChatOllama(
    base_url=os.getenv("LLM_ENDPOINT"),
    model=os.getenv("LLM_MODEL")
)
message = "Explain the theory of relativity in simple terms."
response = client.invoke(message)
print(response.content)
