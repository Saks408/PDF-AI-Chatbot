from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import os
import shutil
import json
import uuid

from datetime import datetime, timezone

from rag import (
    process_pdf,
    ask_question,
    get_documents,
    delete_pdf
)


# =========================================
# APP
# =========================================

app = FastAPI(
    title="PDF AI Chatbot API",
    version="4.0.0"
)


# =========================================
# CORS
# =========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://127.0.0.1:5500",

        "http://localhost:5500"

    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# =========================================
# DIRECTORIES
# =========================================

os.makedirs(
    "uploads",
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)


# =========================================
# CHAT FILE
# =========================================

CHATS_FILE = "data/chats.json"


# =========================================
# MODELS
# =========================================

class QuestionRequest(BaseModel):

    question: str


class NewChatRequest(BaseModel):

    title: str = "New Chat"


# =========================================
# HELPERS
# =========================================

def now_iso():

    return datetime.now(
        timezone.utc
    ).isoformat()


def load_chats():

    if not os.path.exists(
        CHATS_FILE
    ):

        return []

    try:

        with open(
            CHATS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


def save_chats(chats):

    with open(
        CHATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chats,
            f,
            ensure_ascii=False,
            indent=2
        )


def find_chat(
    chats,
    chat_id
):

    for chat in chats:

        if chat["id"] == chat_id:

            return chat

    return None


# =========================================
# HOME
# =========================================

@app.get("/")
def home():

    return {

        "message":
            "PDF AI Chatbot Backend Running",

        "status":
            "online"

    }


# =========================================
# DOCUMENTS
# =========================================

@app.get("/documents")
def documents():

    try:

        docs = get_documents()

        return {

            "documents":
                docs,

            "count":
                len(docs)

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# =========================================
# UPLOAD PDF
# =========================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail="File name is missing."

        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(

            status_code=400,

            detail="Only PDF files are supported."

        )

    filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        "uploads",
        filename
    )

    # -------------------------------------
    # Duplicate check
    # -------------------------------------

    if os.path.exists(
        file_path
    ):

        raise HTTPException(

            status_code=400,

            detail=
                f"'{filename}' is already uploaded."

        )

    try:

        # ---------------------------------
        # Save PDF
        # ---------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ---------------------------------
        # Process PDF
        # ---------------------------------

        result = process_pdf(
            file_path
        )

        return {

            "message":
                "PDF uploaded successfully",

            "filename":
                result["filename"],

            "pages":
                result["pages"],

            "chunks":
                result["chunks"]

        }

    except Exception as e:

        print(
            "\n========== UPLOAD ERROR =========="
        )

        import traceback

        traceback.print_exc()

        print(
            "==================================\n"
        )

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# =========================================
# DELETE PDF
# =========================================

@app.delete("/documents/{filename}")
def remove_pdf(
    filename: str
):

    try:

        result = delete_pdf(
            filename
        )

        return result

    except FileNotFoundError as e:

        raise HTTPException(

            status_code=404,

            detail=str(e)

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# =========================================
# OLD DOCUMENT API
# =========================================

@app.get("/document")
def get_document():

    documents = get_documents()

    if not documents:

        return {

            "uploaded":
                False

        }

    first = documents[0]

    return {

        "uploaded":
            True,

        "filename":
            first["filename"],

        "pages":
            first["pages"],

        "chunks":
            first["chunks"],

        "count":
            len(documents)

    }


# =====================================================
# CHAT APIs
# =====================================================


# =========================================
# GET ALL CHATS
# =========================================

@app.get("/chats")
def get_chats():

    chats = load_chats()

    chats.sort(

        key=lambda x:
            x.get(
                "updated_at",
                ""
            ),

        reverse=True

    )

    return {

        "chats": [

            {

                "id":
                    chat["id"],

                "title":
                    chat.get(
                        "title",
                        "New Chat"
                    ),

                "created_at":
                    chat["created_at"],

                "updated_at":
                    chat["updated_at"],

                "message_count":
                    len(
                        chat.get(
                            "messages",
                            []
                        )
                    )

            }

            for chat in chats

        ]

    }


# =========================================
# CREATE CHAT
# =========================================

@app.post("/chats")
def create_chat(
    request: NewChatRequest
):

    chats = load_chats()

    chat_id = str(
        uuid.uuid4()
    )

    timestamp = now_iso()

    chat = {

        "id":
            chat_id,

        "title":
            request.title,

        "created_at":
            timestamp,

        "updated_at":
            timestamp,

        "messages":
            []

    }

    chats.append(
        chat
    )

    save_chats(
        chats
    )

    return chat


# =========================================
# GET ONE CHAT
# =========================================

@app.get("/chats/{chat_id}")
def get_chat(
    chat_id: str
):

    chats = load_chats()

    chat = find_chat(
        chats,
        chat_id
    )

    if not chat:

        raise HTTPException(

            status_code=404,

            detail="Chat not found."

        )

    return chat


# =========================================
# DELETE CHAT
# =========================================

@app.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str
):

    chats = load_chats()

    chat = find_chat(
        chats,
        chat_id
    )

    if not chat:

        raise HTTPException(

            status_code=404,

            detail="Chat not found."

        )

    chats = [

        chat

        for chat in chats

        if chat["id"] != chat_id

    ]

    save_chats(
        chats
    )

    return {

        "message":
            "Chat deleted successfully"

    }


# =========================================
# CLEAR CHAT
# =========================================

@app.post("/chats/{chat_id}/clear")
def clear_chat(
    chat_id: str
):

    chats = load_chats()

    chat = find_chat(
        chats,
        chat_id
    )

    if not chat:

        raise HTTPException(

            status_code=404,

            detail="Chat not found."

        )

    chat["messages"] = []

    chat["title"] = "New Chat"

    chat["updated_at"] = now_iso()

    save_chats(
        chats
    )

    return chat


# =========================================
# ASK QUESTION
# =========================================

@app.post("/chats/{chat_id}/ask")
def ask_chat_question(

    chat_id: str,

    request: QuestionRequest

):

    question = request.question.strip()

    if not question:

        raise HTTPException(

            status_code=400,

            detail="Question is required."

        )

    # -------------------------------------
    # Check PDF
    # -------------------------------------

    documents = get_documents()

    if not documents:

        raise HTTPException(

            status_code=400,

            detail=
                "Please upload a PDF first."

        )

    # -------------------------------------
    # Find chat
    # -------------------------------------

    chats = load_chats()

    chat = find_chat(
        chats,
        chat_id
    )

    if not chat:

        raise HTTPException(

            status_code=404,

            detail="Chat not found."

        )

    try:

        # =================================
        # RAG + CACHE + VERIFICATION
        # =================================

        
        answer = ask_question(
            question,
            chat["messages"]
        )

        timestamp = now_iso()

        # ---------------------------------
        # User message
        # ---------------------------------

        chat["messages"].append({

            "role":
                "user",

            "content":
                question,

            "created_at":
                timestamp

        })

        # ---------------------------------
        # AI message
        # ---------------------------------

        chat["messages"].append({

            "role":
                "assistant",

            "content":
                answer,

            "created_at":
                now_iso()

        })

        # ---------------------------------
        # First question = title
        # ---------------------------------

        if chat["title"] == "New Chat":

            title = (

                question
                .replace(
                    "\n",
                    " "
                )
                .strip()

            )

            if len(title) > 40:

                title = (
                    title[:40] +
                    "..."
                )

            chat["title"] = title

        chat["updated_at"] = now_iso()

        # ---------------------------------
        # Save chat
        # ---------------------------------

        save_chats(
            chats
        )

        return {

            "chat_id":
                chat_id,

            "title":
                chat["title"],

            "question":
                question,

            "answer":
                answer

        }

    except Exception as e:

        print(
            "Question processing error:",
            e
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )