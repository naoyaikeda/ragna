import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from langchain_core.messages import HumanMessage, AIMessage

class App:
    client = None
    turn_count = 0

    def __init__(self):
        # 環境変数がロードされていることを確認
        load_dotenv()
        self.client = ChatOllama(
            base_url=os.getenv("LLM_ENDPOINT"),
            model=os.getenv("LLM_MODEL")
        )

    def chat(self, message, history): # 引数名は `messages` (最新のユーザー入力)と `history` (履歴)
        # Gradioの履歴をLangChainのMessageオブジェクトに変換
        # history は [["user_msg_1", "llm_response_1"], ["user_msg_2", "llm_response_2"], ...] のリスト
        
        langchain_messages = []
        # ここで history の各要素が [human, ai] のペアであることを期待
        for human, ai in history:
            # human, ai が文字列ではない可能性を考慮して str() を適用しても良いかもしれません
            langchain_messages.append(HumanMessage(content=str(human)))
            langchain_messages.append(AIMessage(content=str(ai)))
    
        # 最新のユーザーメッセージを追加
        # message 引数には最新のユーザー入力 (文字列) が含まれます
        langchain_messages.append(HumanMessage(content=str(message)))
    
        # LangChainのinvokeに完全なメッセージリストを渡す
        response = self.client.invoke(langchain_messages)

        self.turn_count += 1

        return response.content # 返り値は最新のLLMの応答
    
    def run(self):
        # type="messages" を削除しました。fnが (message, history) の2引数を取る場合、これが標準です。
        iface = gr.ChatInterface(fn=self.chat, title="Ragna Chat Interface",
                                 description="Chat with the Ollama language model.")

        # サーバー設定の環境変数処理はそのまま維持
        port = int(os.getenv("PORT", 8833))
        if os.getenv("ALLOW_OUTBOUND", "false").lower() == "true":
            # share=True は外部公開用ですが、server_name="0.0.0.0" と併用します
            iface.launch(server_port=port, server_name="0.0.0.0", share=True)
        else:
            iface.launch(server_port=port)


def main():
    # load_dotenv() は App.__init__ で行われるように移動し、ここでは不要になりましたが、
    # 念のためここに残しておいても害はありません（二重ロードされるだけです）
    load_dotenv() 

    app = App()
    app.run()

if __name__ == "__main__":
    main()
