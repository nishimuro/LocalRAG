import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import (
    get_embeddings,
    load_csv_documents,
    build_vectorstore,
    build_rag_chain,
    CHROMA_PATH
)
from langchain_chroma import Chroma

# ===== グローバル変数 =====
rag_chain = None
retriever = None

# ===== 起動時初期化 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever
    print("=== RAG初期化開始 ===")
    embeddings = get_embeddings()

    # Chromaが既に存在する場合は読み込み、なければ新規作成
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("既存のChromaを読み込み中...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    else:
        print("Chromaを新規作成中...")
        documents = load_csv_documents()
        vectorstore = build_vectorstore(documents, embeddings)

    rag_chain, retriever = build_rag_chain(vectorstore)
    print("=== RAG初期化完了 ===")
    yield

# ===== FastAPIアプリ =====
app = FastAPI(
    title="半導体用語RAG API",
    description="半導体製造の専門用語・テーブル定義を検索するRAG API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== リクエスト/レスポンスモデル =====
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

# ===== エンドポイント =====
@app.get("/")
async def root():
    return {"status": "ok", "message": "半導体用語RAG APIが稼働中です"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="質問が空です")
    try:
        answer = rag_chain.invoke(request.question)
        docs = retriever.invoke(request.question)
        sources = list(set(doc.metadata["source"] for doc in docs))
        return QuestionResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_documents():
    global rag_chain, retriever
    try:
        embeddings = get_embeddings()
        documents = load_csv_documents()
        vectorstore = build_vectorstore(documents, embeddings)
        rag_chain, retriever = build_rag_chain(vectorstore)
        return {"status": "ok", "message": "ドキュメントを再読み込みしました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/ui")
async def ui():
    return FileResponse("/app/static/index.html")
