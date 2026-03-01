import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===== 設定 =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q4_K_M"
EMBEDDING_MODEL = "cl-nagoya/ruri-large"
CHROMA_PATH = "/app/chroma_db"
DOCUMENTS_PATH = "/app/documents"

# ===== Embeddingモデルの初期化 =====
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

# ===== CSVをDocumentに変換 =====
def load_csv_documents():
    documents = []

    # 用語集（パターンA）の読み込み
    terms_path = os.path.join(DOCUMENTS_PATH, "terms.csv")
    if os.path.exists(terms_path):
        df = pd.read_csv(terms_path)
        for _, row in df.iterrows():
            content = f"用語：{row['用語']}\n説明：{row['説明']}"
            doc = Document(
                page_content=content,
                metadata={"source": "terms.csv", "type": "用語集", "term": row['用語']}
            )
            documents.append(doc)
        print(f"用語集：{len(df)}件読み込み完了")

    # テーブル定義書（パターンB）の読み込み
    table_path = os.path.join(DOCUMENTS_PATH, "table_definitions.csv")
    if os.path.exists(table_path):
        df = pd.read_csv(table_path)
        for table_name, group in df.groupby("テーブル名"):
            lines = [f"テーブル名：{table_name}"]
            lines.append(f"工程：{group['工程'].iloc[0]}")
            for _, row in group.iterrows():
                lines.append(
                    f"列名：{row['列名']} / 意味：{row['意味']} / "
                    f"正常範囲：{row['正常範囲']}{row['単位']} / 備考：{row['備考']}"
                )
            content = "\n".join(lines)
            doc = Document(
                page_content=content,
                metadata={"source": "table_definitions.csv", "type": "テーブル定義", "table": table_name}
            )
            documents.append(doc)
        print(f"テーブル定義：{df['テーブル名'].nunique()}テーブル読み込み完了")

    return documents

# ===== Chromaへの登録 =====
def build_vectorstore(documents, embeddings):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Chroma登録完了：{len(documents)}件")
    return vectorstore

# ===== プロンプトテンプレート =====
PROMPT_TEMPLATE = """あなたは半導体製造の専門知識を持つアシスタントです。
以下の参考情報をもとに、質問に日本語で答えてください。
参考情報に答えがない場合は「情報が見つかりませんでした」と答えてください。

参考情報：
{context}

質問：{question}

回答："""

# ===== コンテキスト整形 =====
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ===== RAGチェーンの構築 =====
def build_rag_chain(vectorstore):
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ===== メイン処理 =====
if __name__ == "__main__":
    print("=== RAGパイプライン初期化 ===")
    embeddings = get_embeddings()
    documents = load_csv_documents()
    vectorstore = build_vectorstore(documents, embeddings)
    chain, retriever = build_rag_chain(vectorstore)

    print("\n=== 動作確認 ===")
    questions = [
        "フォトリソグラフィとは何ですか？",
        "inspection_photoテーブルはどのような情報を持っていますか？",
        "line_widthの正常範囲を教えてください。"
    ]
    for q in questions:
        print(f"\n質問：{q}")
        result = chain.invoke(q)
        print(f"回答：{result}")
        docs = retriever.invoke(q)
        print(f"参照元：{[doc.metadata['source'] for doc in docs]}")
