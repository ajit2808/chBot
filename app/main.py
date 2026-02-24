import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.config import settings
from app.ingestion import ingest_file
from app.chunking import chunk_texts
from app.embeddings import generate_embeddings
from app.vector_store import add_documents
from app.retriever import retrieve
from app.claude_client import generate_answer

app = FastAPI(title="Enterprise RAG System")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processing_status = {"progress": 0, "status": "idle"}

@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/progress/")
async def get_progress():
    return processing_status

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):

    size_mb = file.size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_MB:
        return {"error": f"File exceeds {settings.MAX_FILE_MB}MB limit."}

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_file, file_path)

    return {"message": "File uploaded. Processing started."}

def process_file(file_path):
    processing_status["status"] = "processing"
    processing_status["progress"] = 10

    texts = ingest_file(file_path)
    processing_status["progress"] = 40

    chunks = chunk_texts(texts)
    processing_status["progress"] = 60

    embeddings = generate_embeddings(chunks)
    processing_status["progress"] = 80

    add_documents(chunks, embeddings)
    processing_status["progress"] = 100
    processing_status["status"] = "complete"

class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask(payload: Question):
    try:
        docs = retrieve(payload.question, top_k=15)

        context = "\n\n".join(docs[:10])  # prevent token overload

        answer = generate_answer(context, payload.question)

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}