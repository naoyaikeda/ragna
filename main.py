import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

client = ChatOllama(
    base_url=os.getenv("LLM_ENDPOINT"),
    model=os.getenv("LLM_MODEL")
)

def chat(messages, history):
# Gradioの履歴をLangChainのMessageオブジェクトに変換
    # history は [["user_msg_1", "llm_response_1"], ["user_msg_2", "llm_response_2"], ...] のリスト
    langchain_messages = []
    for human, ai in history:
        langchain_messages.append(HumanMessage(content=human))
        langchain_messages.append(AIMessage(content=ai))
    
    # 最新のユーザーメッセージを追加
    langchain_messages.append(HumanMessage(content=messages))
    
    # LangChainのinvokeに完全なメッセージリストを渡す
    response = client.invoke(langchain_messages)
    
    return response.content # 返り値は最新のLLMの応答
iface = gr.ChatInterface(fn=chat, title="Ragna Chat Interface",
                         description="Chat with the Ollama language model.", type="messages")

if os.getenv("ALLOW_OUTBOUND", "false").lower() == "true":
    iface.launch(server_port=int(os.getenv("PORT", 8833)), server_name="0.0.0.0")
else:
    iface.launch(server_port=int(os.getenv("PORT", 8833)))
