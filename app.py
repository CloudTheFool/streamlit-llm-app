import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm_response(question: str, expert_type: str) -> str:
    if expert_type == "A: 医学専門家":
        system_message = "あなたは医学の専門家です。医学的な観点で、簡潔かつ正確に回答してください。"
    else:
        system_message = "あなたは経済の専門家です。経済的な観点で、簡潔かつ正確に回答してください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


st.title("LLM Expert Chat App")

expert_type = st.radio("専門家の種類を選択してください", ("A: 医学専門家", "B: 経済専門家"))
user_input = st.text_input("質問を入力してください")

if st.button("送信"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY が未設定です。.env に設定してください。")
    elif not user_input.strip():
        st.warning("質問を入力してください。")
    else:
        response = get_llm_response(user_input, expert_type)
        st.write(response)

#Webアプリの概要や操作方法をユーザーに明示するためのテキストを表示してください。
st.markdown(
    """
    このアプリは、医学専門家または経済専門家として振る舞うLLM（大規模言語モデル）に質問を送信し、専門的な回答を得ることができます。
    使い方:
    1. 専門家の種類を選択してください。
    2. 質問を入力してください。
    3. 「送信」ボタンを押して回答を取得してください。
    """
)