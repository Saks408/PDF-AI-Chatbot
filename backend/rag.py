import os
import json
import time
import re

import numpy as np
import faiss

from pypdf import PdfReader
from openai import OpenAI
from openai import InternalServerError, APIConnectionError
from dotenv import load_dotenv

from verified_cache import (
    find_cached_answer,
    store_verified_answer,
    clear_cache
)


# =========================================
# ENVIRONMENT
# =========================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY environment variable is not set."
    )


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


# =========================================
# MODELS
# =========================================

EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


# =========================================
# PATHS
# =========================================

DATA_DIR = "data"

FAISS_PATH = os.path.join(
    DATA_DIR,
    "database.faiss"
)

CHUNKS_PATH = os.path.join(
    DATA_DIR,
    "chunks.json"
)

DOCUMENTS_PATH = os.path.join(
    DATA_DIR,
    "documents.json"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# =========================================
# CACHE
# =========================================

CACHE_SIMILARITY_THRESHOLD = 0.75


# =========================================
# FALLBACK MESSAGE
# =========================================

NOT_FOUND_MESSAGE = (
    "I could not find the answer in the provided document."
)

# =========================================
# CHUNKING - RECURSIVE
# =========================================

CHUNK_SIZE = 150
CHUNK_OVERLAP = 20


def _recursive_split(
    text,
    max_words,
    separators=None
):
    """
    Recursively split text using meaningful boundaries.

    Priority:
    1. Paragraph
    2. Line
    3. Sentence
    4. Clause
    5. Space
    """

    if separators is None:
        separators = [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " "
        ]

    text = text.strip()

    if not text:
        return []

    # Already small enough
    if len(text.split()) <= max_words:
        return [text]

    # Find the first separator that exists
    separator = None

    for sep in separators:

        if sep in text:
            separator = sep
            break

    # No separator found
    # -> fallback to word splitting
    if separator is None:

        words = text.split()

        return [
            " ".join(
                words[i:i + max_words]
            )
            for i in range(
                0,
                len(words),
                max_words
            )
        ]

    # Split using selected separator
    if separator == " ":

        pieces = text.split()

    else:

        pieces = text.split(separator)

    pieces = [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]

    # If splitting didn't actually happen,
    # try the next separator
    if len(pieces) <= 1:

        remaining_separators = separators[1:]

        if not remaining_separators:

            words = text.split()

            return [
                " ".join(
                    words[i:i + max_words]
                )
                for i in range(
                    0,
                    len(words),
                    max_words
                )
            ]

        return _recursive_split(
            text,
            max_words,
            remaining_separators
        )

    result = []

    for piece in pieces:

        if len(piece.split()) <= max_words:

            result.append(piece)

        else:

            result.extend(
                _recursive_split(
                    piece,
                    max_words,
                    separators[1:]
                )
            )

    return result


def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Recursive chunking.

    Instead of blindly cutting every 150 words,
    this tries to preserve:

    paragraph -> line -> sentence -> clause

    while keeping chunk size around 250 words
    with 50 words overlap.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if (
        chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be >= 0 "
            "and less than chunk_size"
        )

    # Normalize line endings
    text = re.sub(
        r"\r\n?",
        "\n",
        text
    ).strip()

    # Get meaningful pieces
    pieces = _recursive_split(
        text,
        chunk_size
    )

    if not pieces:
        return []

    chunks = []
    current_words = []

    for piece in pieces:

        words = piece.split()

        if not words:
            continue

        # Safety fallback
        if len(words) > chunk_size:

            if current_words:

                chunks.append(
                    " ".join(
                        current_words
                    ).strip()
                )

                current_words = (
                    current_words[
                        -chunk_overlap:
                    ]
                    if chunk_overlap
                    else []
                )

            step = (
                chunk_size - chunk_overlap
            )

            for start in range(
                0,
                len(words),
                step
            ):

                part = words[
                    start:start + chunk_size
                ]

                if part:

                    chunks.append(
                        " ".join(part).strip()
                    )

            current_words = []

            continue

        # Add piece to current chunk
        if (
            len(current_words)
            + len(words)
            <= chunk_size
        ):

            current_words.extend(
                words
            )

        else:

            # Save current chunk
            if current_words:

                chunks.append(
                    " ".join(
                        current_words
                    ).strip()
                )

            # Add overlap
            overlap_words = (
                current_words[
                    -chunk_overlap:
                ]
                if chunk_overlap
                else []
            )

            current_words = (
                overlap_words + words
            )

            # Safety trim
            if (
                len(current_words)
                > chunk_size
            ):

                current_words = (
                    current_words[
                        -chunk_size:
                    ]
                )

    # Add final chunk
    if current_words:

        chunks.append(
            " ".join(
                current_words
            ).strip()
        )

    # Clean duplicates
    final_chunks = []

    seen = set()

    for chunk in chunks:

        chunk = re.sub(
            r"\s+",
            " ",
            chunk
        ).strip()

        if not chunk:
            continue

        if chunk in seen:
            continue

        seen.add(chunk)

        final_chunks.append(chunk)

    return final_chunks


# =========================================
# CREATE DOCUMENT EMBEDDINGS
# =========================================

def create_embeddings(
    chunks,
    batch_size=10
):

    if not chunks:
        return []

    embeddings = []

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            i:i + batch_size
        ]

        print(
            f"Creating embeddings: "
            f"{i + 1} - "
            f"{min(i + batch_size, len(chunks))} "
            f"/ {len(chunks)}"
        )

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            extra_body={
                "input_type": "passage"
            }
        )

        batch_data = sorted(
            response.data,
            key=lambda item: item.index
        )

        for item in batch_data:

            embeddings.append(
                item.embedding
            )

    return embeddings


# =========================================
# CREATE QUERY EMBEDDING
# =========================================

def create_query_embedding(query):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
        extra_body={
            "input_type": "query"
        }
    )

    embedding = response.data[0].embedding

    print(
        f"Query embedding dimension: "
        f"{len(embedding)}"
    )

    return embedding


# =========================================
# LOAD CHUNKS
# =========================================

def load_chunks():

    if not os.path.exists(CHUNKS_PATH):
        return []

    try:

        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception as e:

        print(
            "Chunk load error:",
            e
        )

        return []


# =========================================
# SAVE CHUNKS
# =========================================

def save_chunks(chunks):

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================================
# LOAD DOCUMENTS
# =========================================

def load_documents():

    if not os.path.exists(DOCUMENTS_PATH):
        return []

    try:

        with open(
            DOCUMENTS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception as e:

        print(
            "Document load error:",
            e
        )

        return []


# =========================================
# SAVE DOCUMENTS
# =========================================

def save_documents(documents):

    with open(
        DOCUMENTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================================
# REBUILD FAISS
# =========================================

def rebuild_faiss():

    chunks = load_chunks()

    if not chunks:

        if os.path.exists(FAISS_PATH):
            os.remove(FAISS_PATH)

        print(
            "No chunks available. FAISS removed."
        )

        return

    valid_items = [
        item
        for item in chunks
        if item.get("text")
    ]

    if not valid_items:

        if os.path.exists(FAISS_PATH):
            os.remove(FAISS_PATH)

        return

    texts = [
        item["text"]
        for item in valid_items
    ]

    embeddings = create_embeddings(texts)

    if not embeddings:
        raise RuntimeError(
            "Failed to create embeddings."
        )

    matrix = np.array(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(matrix)

    dimension = matrix.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(matrix)

    faiss.write_index(
        index,
        FAISS_PATH
    )

    print(
        f"FAISS index rebuilt successfully. "
        f"Vectors: {index.ntotal}, "
        f"Dimension: {dimension}"
    )


# =========================================
# PROCESS PDF
# =========================================

def process_pdf(pdf_path):

    filename = os.path.basename(
        pdf_path
    )

    reader = PdfReader(
        pdf_path
    )

    total_pages = len(
        reader.pages
    )

    print(
        f"Processing PDF: {filename}"
    )

    all_text = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            page_text = page.extract_text()

        except Exception as e:

            print(
                f"Page {page_number} "
                f"text extraction failed:",
                e
            )

            page_text = None

        if page_text:

            all_text.append(
                page_text
            )

    combined_text = "\n\n".join(
       all_text
   )

    combined_text = re.sub(
        r"[ \t]+",
        " ",
        combined_text
)

    combined_text = re.sub(
        r"\n{3,}",
        "\n\n",
        combined_text
    ).strip()

    if not combined_text:

        raise ValueError(
            "No extractable text was found "
            "in this PDF. It may be a scanned "
            "or image-only PDF."
        )

    new_chunks = create_chunks(
        combined_text,
        chunk_size=150,
        chunk_overlap=20
    )

    if not new_chunks:

        raise ValueError(
            "No chunks were created from this PDF."
        )

    print(
        f"Chunks created: {len(new_chunks)}"
    )

    existing_chunks = load_chunks()


    # =====================================
    # DUPLICATE PDF CHECK
    # =====================================

    for item in existing_chunks:

        if item.get("filename") == filename:

            raise ValueError(
                f"'{filename}' is already uploaded."
            )


    starting_id = len(
        existing_chunks
    )

    for index, chunk in enumerate(
        new_chunks
    ):

        existing_chunks.append({

            "id":
                starting_id + index,

            "filename":
                filename,

            "text":
                chunk
        })

    save_chunks(
        existing_chunks
    )

    try:

        rebuild_faiss()

    except Exception:

        rolled_back = [

            item

            for item in existing_chunks

            if item.get("filename") != filename
        ]

        save_chunks(
            rolled_back
        )

        raise


    documents = load_documents()

    documents.append({

        "filename":
            filename,

        "pages":
            total_pages,

        "chunks":
            len(new_chunks)
    })

    save_documents(
        documents
    )


    # =====================================
    # CLEAR OLD ANSWER CACHE
    # =====================================

    clear_cache()

    print(
        f"PDF added successfully: {filename}"
    )

    return {

        "filename":
            filename,

        "pages":
            total_pages,

        "chunks":
            len(new_chunks)
    }

# =========================================
# REBUILD DATABASE FROM UPLOADED PDFs
# =========================================

def rebuild_database_from_uploads():

    uploads_dir = "uploads"

    if not os.path.exists(uploads_dir):

        print(
            "Uploads directory not found."
        )

        return

    pdf_files = [
        filename
        for filename in os.listdir(uploads_dir)
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print(
            "No PDF files found in uploads directory."
        )

        # Clear old database
        save_chunks([])
        save_documents([])

        if os.path.exists(FAISS_PATH):
            os.remove(FAISS_PATH)

        clear_cache()

        print(
            "Database cleared."
        )

        return

    print(
        f"Found {len(pdf_files)} PDF file(s)."
    )

    print(
        "Rebuilding database using recursive chunking..."
    )

    # =====================================
    # CLEAR OLD DATABASE
    # =====================================

    save_chunks([])
    save_documents([])

    if os.path.exists(FAISS_PATH):

        os.remove(FAISS_PATH)

    clear_cache()

    # =====================================
    # PROCESS ALL PDFs
    # =====================================

    successful = 0
    failed = 0

    for filename in pdf_files:

        pdf_path = os.path.join(
            uploads_dir,
            filename
        )

        print(
            "\n========================================"
        )

        print(
            f"Processing: {filename}"
        )

        print(
            "========================================"
        )

        try:

            process_pdf(pdf_path)

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"Failed to process {filename}: {e}"
            )

    # =====================================
    # FINAL STATUS
    # =====================================

    print(
        "\n========================================"
    )

    print(
        "DATABASE REBUILD COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Successful PDFs: {successful}"
    )

    print(
        f"Failed PDFs: {failed}"
    )

    print(
        f"Total chunks: {len(load_chunks())}"
    )

    print(
        f"Total documents: {len(load_documents())}"
    )

    print(
        "========================================"
    )
# =========================================
# GET DOCUMENTS
# =========================================

def get_documents():

    return load_documents()


# =========================================
# DELETE PDF
# =========================================

def delete_pdf(filename):

    chunks = load_chunks()

    matching_chunks = [

        item

        for item in chunks

        if item.get("filename") == filename
    ]

    if not matching_chunks:

        raise FileNotFoundError(
            f"PDF '{filename}' not found."
        )


    remaining_chunks = [

        item

        for item in chunks

        if item.get("filename") != filename
    ]


    for index, item in enumerate(
        remaining_chunks
    ):

        item["id"] = index


    save_chunks(
        remaining_chunks
    )


    documents = load_documents()

    documents = [

        doc

        for doc in documents

        if doc.get("filename") != filename
    ]

    save_documents(
        documents
    )


    pdf_path = os.path.join(
        "uploads",
        filename
    )

    if os.path.exists(pdf_path):

        os.remove(
            pdf_path
        )


    rebuild_faiss()

    clear_cache()

    print(
        f"PDF deleted: {filename}"
    )

    return {

        "message":
            f"{filename} deleted successfully."
    }


# =========================================
# CLEAN MODEL OUTPUT
# =========================================

def clean_model_answer(answer):

    if not answer:
        return ""

    answer = answer.strip()


    # Remove thinking tags
    if "<think>" in answer:

        if "</think>" in answer:

            answer = answer.split(
                "</think>",
                1
            )[1].strip()

        else:

            answer = answer.replace(
                "<think>",
                ""
            ).strip()


    prefixes = [
        "Final answer:",
        "Final Answer:",
        "Answer:",
        "Response:",
        "Final:"
    ]


    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()


    return answer


# =========================================
# CALL LLM
# =========================================

def call_llm(
    system_prompt,
    user_prompt,
    max_tokens=400
):

    max_retries = 5

    for attempt in range(
        max_retries
    ):

        try:

            print(
                f"Calling LLM... "
                f"Attempt {attempt + 1}/"
                f"{max_retries}"
            )

            response = client.chat.completions.create(

                model=CHAT_MODEL,

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            system_prompt
                    },

                    {
                        "role":
                            "user",

                        "content":
                            user_prompt
                    }
                ],

                temperature=0.0,

                max_tokens=max_tokens
            )


            answer = (
                response
                .choices[0]
                .message
                .content
            )


            if answer is None:

                raise RuntimeError(
                    "LLM returned empty content."
                )


            answer = clean_model_answer(
                answer
            )


            if not answer:

                raise RuntimeError(
                    "LLM returned empty answer."
                )


            return answer


        except (
            InternalServerError,
            APIConnectionError
        ) as e:

            print(
                f"LLM error: {e}"
            )


            if attempt < max_retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            else:

                raise RuntimeError(
                    f"LLM request failed: {e}"
                )


        except Exception as e:

            error_text = str(e)


            # NVIDIA 503 overload
            if (
                "503" in error_text
                or
                "overloaded" in error_text.lower()
                or
                "Service Unavailable" in error_text
            ):

                print(
                    f"LLM service temporarily unavailable: "
                    f"{error_text}"
                )


                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue


            print(
                f"Unexpected LLM error: {e}"
            )

            raise RuntimeError(
                f"LLM request failed: {e}"
            )


    raise RuntimeError(
        "LLM request failed."
    )


# =========================================
# BUILD CONTEXT
# =========================================

def build_context(
    retrieved_chunks
):

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(

            f"DOCUMENT: "
            f"{item.get('filename', 'Unknown')}\n\n"

            f"CONTENT:\n"
            f"{item.get('text', '')}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================================
# CHECK WHETHER QUESTION IS FOLLOW-UP
# =========================================

def is_follow_up_question(
    query,
    chat_history
):

    if not chat_history:
        return False


    # =====================================
    # IMPORTANT FIX
    # =====================================
    #
    # Previously:
    #
    # any(word in query_lower for word in words)
    #
    # This caused:
    #
    # "benefits" -> contains "it"
    #
    # Therefore standalone questions were
    # incorrectly detected as follow-ups.
    #
    # Now we check COMPLETE WORDS only.
    # =====================================


    follow_up_words = {
        "above",
        "previous",
        "earlier",
        "mentioned",
        "that",
        "it",
        "they",
        "them",
        "those",
        "these",
        "same",
        "also",
        "add",
        "total",
        "calculate",
        "then",
        "this",
        "its",
        "their"
    }


    query_lower = query.lower()


    # Extract actual words from query
    query_words = set(
        re.findall(
            r"\b[\w'-]+\b",
            query_lower
        )
    )


    matched_words = (
        query_words
        & follow_up_words
    )


    if matched_words:

        print(
            "Follow-up indicators found:",
            sorted(matched_words)
        )

        return True


    # =====================================
    # Additional phrase detection
    # =====================================

    follow_up_phrases = [
        "what about",
        "how about",
        "and what about",
        "what about the",
        "what about this",
        "what about that",
        "as mentioned above",
        "mentioned above",
        "previous answer",
        "previous question",
        "earlier answer",
        "earlier question",
        "according to what you said",
        "based on your previous answer"
    ]


    for phrase in follow_up_phrases:

        if phrase in query_lower:

            print(
                "Follow-up phrase found:",
                phrase
            )

            return True


    return False


# =========================================
# REWRITE FOLLOW-UP QUESTION
# =========================================

def rewrite_question_with_history(
    query,
    chat_history
):

    if not chat_history:
        return query


    recent_history = chat_history[-10:]


    history_parts = []


    for message in recent_history:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        ).strip()


        if not content:
            continue


        if role == "user":

            history_parts.append(
                f"USER: {content}"
            )


        elif role == "assistant":

            history_parts.append(
                f"ASSISTANT: {content}"
            )


    if not history_parts:

        return query


    history_text = "\n".join(
        history_parts
    )


    system_prompt = """
You rewrite follow-up questions.

Your ONLY job is to rewrite the CURRENT QUESTION
into one standalone question.

Use the conversation history ONLY when the
CURRENT QUESTION refers to something from it.

Examples of references:

- above
- previous
- earlier
- mentioned
- that
- it
- they
- them
- those
- these
- this
- same
- what about
- how about
- the policy mentioned earlier
- the leave mentioned above

IMPORTANT:

1. Do NOT answer the question.
2. Do NOT explain anything.
3. Do NOT describe your reasoning.
4. Do NOT mention the user.
5. Do NOT mention conversation history.
6. Return ONLY the rewritten question.
7. If the question is already standalone,
   return it EXACTLY as it is.

Example:

USER:
How many leave days are available?

ASSISTANT:
Employees get 10 days of leave.

CURRENT QUESTION:
Can you calculate it for two months?

OUTPUT:
Can you calculate the 10 days of leave for two months?

Another example:

USER:
What is the notice period?

ASSISTANT:
The notice period is 30 days.

CURRENT QUESTION:
What about that for two months?

OUTPUT:
What about the 30-day notice period for two months?

Remember:
NEVER answer the question.
ONLY rewrite the question.
"""


    user_prompt = f"""
CONVERSATION HISTORY:

{history_text}

CURRENT QUESTION:

{query}

Return ONLY the rewritten standalone question.
"""


    try:

        rewritten = call_llm(
            system_prompt,
            user_prompt,
            max_tokens=250
        )


        rewritten = clean_model_answer(
            rewritten
        ).strip()


        if rewritten:

            print(
                "Rewritten question:",
                rewritten
            )

            return rewritten


    except Exception as e:

        print(
            "Question rewrite failed:",
            e
        )


        # IMPORTANT:
        # If rewrite fails, use original
        # question instead of crashing.

        print(
            "Using original question."
        )


    return query


# =========================================
# ASK QUESTION
# =========================================

# =========================================
# ASK QUESTION - IMPROVED RETRIEVAL
# =========================================

def ask_question(
    query,
    chat_history=None
):

    query = query.strip()

    if not query:
        raise ValueError(
            "Question cannot be empty."
        )

    if chat_history is None:
        chat_history = []

    # =====================================
    # CHECK DATABASE
    # =====================================

    if not os.path.exists(FAISS_PATH):

        raise FileNotFoundError(
            "No PDF has been uploaded yet."
        )

    if not os.path.exists(CHUNKS_PATH):

        raise FileNotFoundError(
            "Chunk database not found."
        )

    chunks = load_chunks()

    if not chunks:

        raise FileNotFoundError(
            "No PDF data available."
        )

    # =====================================
    # DETECT FOLLOW-UP
    # =====================================

    looks_like_follow_up = (
        is_follow_up_question(
            query,
            chat_history
        )
    )

    # =====================================
    # REWRITE QUESTION
    # =====================================

    search_query = query

    if looks_like_follow_up:

        print(
            "Follow-up question detected."
        )

        search_query = (
            rewrite_question_with_history(
                query,
                chat_history
            )
        )

    else:

        print(
            "Standalone question detected."
        )

    # =====================================
    # QUERY EMBEDDING
    # =====================================

    print(
        "Creating query embedding..."
    )

    query_embedding = (
        create_query_embedding(
            search_query
        )
    )

    # =====================================
    # CACHE
    # =====================================

    if not looks_like_follow_up:

        print(
            "Checking answer cache..."
        )

        cached_result = (
            find_cached_answer(
                query,
                query_embedding,
                threshold=CACHE_SIMILARITY_THRESHOLD
            )
        )

        if cached_result:

            print(
                "CACHE HIT - "
                f"Matched question: "
                f"{cached_result['question']}"
            )

            return cached_result["answer"]

        print(
            "CACHE MISS"
        )

    else:

        print(
            "Skipping cache because this "
            "is a history-dependent question."
        )

    # =====================================
    # LOAD FAISS
    # =====================================

    index = faiss.read_index(
        FAISS_PATH
    )

    if index.ntotal == 0:

        raise ValueError(
            "FAISS index is empty."
        )

    # =====================================
    # DIMENSION CHECK
    # =====================================

    if len(query_embedding) != index.d:

        raise ValueError(
            "Embedding dimension mismatch. "
            f"Query dimension: "
            f"{len(query_embedding)}, "
            f"FAISS dimension: {index.d}."
        )

    # =====================================
    # QUERY VECTOR
    # =====================================

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )

    faiss.normalize_L2(
        query_vector
    )

    # =====================================
    # IMPROVED RETRIEVAL
    # =====================================
    #
    # Step 1:
    # Retrieve more candidates.
    #
    # Step 2:
    # Remove weak similarity results.
    #
    # Step 3:
    # Sort by similarity score.
    #
    # Step 4:
    # Send only the best chunks to LLM.
    # =====================================

    candidate_k = min(
        8,
        len(chunks),
        index.ntotal
    )

    print(
        f"\nSearching FAISS with: "
        f"{search_query}"
    )

    print(
        f"Retrieving top "
        f"{candidate_k} candidate chunks..."
    )

    distances, indices = index.search(
        query_vector,
        candidate_k
    )

    # =====================================
    # COLLECT CANDIDATES
    # =====================================

    retrieved_chunks = []

    for position, idx in enumerate(
        indices[0]
    ):

        idx = int(idx)

        if 0 <= idx < len(chunks):

            item = chunks[idx].copy()

            item["score"] = float(
                distances[0][position]
            )

            retrieved_chunks.append(
                item
            )

            print(
                "Candidate chunk: "
                f"filename={item.get('filename')}, "
                f"chunk_id={item.get('id')}, "
                f"score={item['score']:.4f}"
            )

    # =====================================
    # NO RETRIEVAL
    # =====================================

    if not retrieved_chunks:

        print(
            "No chunks retrieved."
        )

        return NOT_FOUND_MESSAGE

    # =====================================
    # SIMILARITY THRESHOLD
    # =====================================

    RETRIEVAL_THRESHOLD = 0.30

    relevant_chunks = [

        item

        for item in retrieved_chunks

        if item["score"] >= RETRIEVAL_THRESHOLD
    ]

    print(
        f"\nChunks above similarity "
        f"threshold {RETRIEVAL_THRESHOLD}: "
        f"{len(relevant_chunks)}"
    )

    # =====================================
    # SORT BY SIMILARITY
    # =====================================

    relevant_chunks = sorted(
        relevant_chunks,
        key=lambda item: item["score"],
        reverse=True
    )

    # =====================================
    # KEEP BEST CHUNKS
    # =====================================

    MAX_CONTEXT_CHUNKS = 4

    relevant_chunks = (
        relevant_chunks[
            :MAX_CONTEXT_CHUNKS
        ]
    )

    # =====================================
    # FINAL RETRIEVED CHUNKS
    # =====================================

    print(
        "\n========== FINAL RETRIEVED CHUNKS =========="
    )

    for item in relevant_chunks:

        print(
            f"filename={item.get('filename')}, "
            f"chunk_id={item.get('id')}, "
            f"score={item['score']:.4f}"
        )

    print(
        "============================================\n"
    )

    # =====================================
    # NO SUFFICIENTLY RELEVANT CHUNKS
    # =====================================

    if not relevant_chunks:

        print(
            "No sufficiently relevant chunks found."
        )

        return NOT_FOUND_MESSAGE

    # =====================================
    # BUILD CONTEXT
    # =====================================

    context = build_context(
        relevant_chunks
    )

    # =====================================
    # BUILD CHAT HISTORY
    # =====================================

    history_text = ""

    recent_history = chat_history[-10:]

    if recent_history:

        history_parts = []

        for message in recent_history:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            ).strip()

            if not content:
                continue

            if role == "user":

                history_parts.append(
                    f"USER: {content}"
                )

            elif role == "assistant":

                history_parts.append(
                    f"ASSISTANT: {content}"
                )

        history_text = "\n".join(
            history_parts
        )

    # =====================================
    # SYSTEM PROMPT
    # =====================================

    system_prompt = """
You are a PDF question-answering assistant.

Answer the CURRENT QUESTION using the SOURCE TEXT.

You may use CONVERSATION HISTORY only when
the current question is a genuine follow-up
question.

IMPORTANT RULES:

1. Use information from SOURCE TEXT whenever
   the answer is available there.

2. Use conversation history only to resolve
   references such as:
   "it", "that", "those", "above", "previous",
   "mentioned earlier", "what about", etc.

3. Do not invent facts.

4. Do not assume information that is not
   present in the source text or conversation.

5. If the answer cannot be found in the
   source text or required information is
   unavailable, return exactly:

I could not find the answer in the provided document.

6. Perform simple calculations when required
   and when all required numbers are available.

7. Return ONLY the final answer.

8. Do NOT show reasoning.

9. Do NOT mention:
   - RAG
   - FAISS
   - embeddings
   - chunks
   - retrieval
   - source text
   - conversation history
   - question rewriting
   - internal processing

10. Do NOT say:
   "The user is asking..."
   "Let me check..."
   "Let me analyze..."
   "Based on the retrieval..."
   "According to the context..."

Answer the user's actual question directly.
"""

    # =====================================
    # USER PROMPT
    # =====================================

    user_prompt = f"""
CONVERSATION HISTORY:

{history_text if history_text else "No previous conversation."}


SOURCE TEXT:

{context}


ORIGINAL CURRENT QUESTION:

{query}


SEARCH QUESTION:

{search_query}


Answer ONLY the ORIGINAL CURRENT QUESTION.
"""

    # =====================================
    # DEBUG CONTEXT
    # =====================================

    print(
        "\n================ CONTEXT SENT TO LLM ================\n"
    )

    print(
        context
    )

    print(
        "\n======================================================\n"
    )

    # =====================================
    # CALL LLM
    # =====================================

    print(
        "Calling LLM..."
    )

    generated_answer = call_llm(
        system_prompt,
        user_prompt,
        max_tokens=400
    )

    if not generated_answer:

        return NOT_FOUND_MESSAGE

    # =====================================
    # CLEAN ANSWER
    # =====================================

    generated_answer = (
        clean_model_answer(
            generated_answer
        )
        .strip()
    )

    print(
        "LLM answer:",
        repr(generated_answer)
    )

    # =====================================
    # NOT FOUND
    # =====================================

    if (
        generated_answer.lower().strip()
        ==
        NOT_FOUND_MESSAGE.lower().strip()
    ):

        print(
            "LLM says answer was not found."
        )

        return NOT_FOUND_MESSAGE

    # =====================================
    # REMOVE PREFIX
    # =====================================

    unwanted_prefixes = [
        "answer:",
        "final answer:",
        "response:",
        "final:"
    ]

    answer_lower = (
        generated_answer.lower()
    )

    for prefix in unwanted_prefixes:

        if answer_lower.startswith(prefix):

            generated_answer = (
                generated_answer[
                    len(prefix):
                ].strip()
            )

            break

    # =====================================
    # REASONING CHECK
    # =====================================

    reasoning_patterns = [

        "the user is asking",
        "the user wants",
        "i need to look",
        "i need to determine",
        "i need to provide",
        "let me check",
        "let me analyze",
        "the question asks",
        "key points from",
        "based on the context",
        "based on the document",
        "the current question",
        "conversation history shows"
    ]

    answer_lower = (
        generated_answer.lower()
    )

    if any(
        pattern in answer_lower[:300]
        for pattern in reasoning_patterns
    ):

        print(
            "Invalid reasoning response."
        )

        return NOT_FOUND_MESSAGE

    # =====================================
    # FINAL EMPTY CHECK
    # =====================================

    if not generated_answer:

        return NOT_FOUND_MESSAGE

    # =====================================
    # STORE CACHE
    # =====================================

    if not looks_like_follow_up:

        store_verified_answer(
            query,
            generated_answer,
            query_embedding
        )

        print(
            "Answer stored in cache."
        )

    else:

        print(
            "Follow-up answer NOT stored "
            "in global answer cache."
        )

    # =====================================
    # RETURN
    # =====================================

    return generated_answer