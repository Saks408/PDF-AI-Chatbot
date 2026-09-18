# Enhanced PDF AI Chatbot

## Folder structure

frontend/
    index.html
    style.css
    script.js

backend/
    main.py
    rag.py
    requirements.txt
    uploads/
    data/

## Backend setup

Open a terminal inside `backend`:

```bash
pip install -r requirements.txt
```

Set your NVIDIA API key.

PowerShell:

```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

Then start FastAPI:

```bash
uvicorn main:app --reload
```

Backend:
http://127.0.0.1:8000

## Frontend

Open the `frontend` folder with VS Code Live Server.

The frontend is configured for:

http://127.0.0.1:5500

## Important

The original `rag.py` had an indentation problem:
the embedding API call was outside the batching loop.

That has been fixed in this version.

The new UI includes:
- Chatbot-style conversation
- PDF drag & drop
- Upload progress/status
- Current document information
- Suggested questions
- Enter-to-send
- Loading/typing indicator
- Clear chat
- New chat
- Responsive mobile layout
- Better error handling
