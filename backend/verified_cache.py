import os
import json
import numpy as np


# =========================================
# CACHE FILE
# =========================================

CACHE_FILE = os.path.join(
    "data",
    "answer_cache.json"
)

os.makedirs(
    "data",
    exist_ok=True
)


# =========================================
# CACHE THRESHOLD
# =========================================

CACHE_SIMILARITY_THRESHOLD = 0.75


# =========================================
# LOAD CACHE
# =========================================

def load_cache():

    if not os.path.exists(CACHE_FILE):
        return []

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception as e:

        print("Cache load error:", e)

        return []


# =========================================
# SAVE CACHE
# =========================================

def save_cache(cache):

    try:

        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                cache,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print("Cache save error:", e)


# =========================================
# CLEAR CACHE
# =========================================

def clear_cache():

    if os.path.exists(CACHE_FILE):

        os.remove(CACHE_FILE)

        print("Answer cache cleared.")


# =========================================
# COSINE SIMILARITY
# =========================================

def cosine_similarity(
    vector1,
    vector2
):

    try:

        v1 = np.asarray(
            vector1,
            dtype="float32"
        )

        v2 = np.asarray(
            vector2,
            dtype="float32"
        )

        if v1.ndim != 1:
            v1 = v1.flatten()

        if v2.ndim != 1:
            v2 = v2.flatten()

        if len(v1) != len(v2):

            print(
                "Embedding dimension mismatch:",
                len(v1),
                len(v2)
            )

            return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(
            np.dot(v1, v2)
            /
            (norm1 * norm2)
        )

    except Exception as e:

        print(
            "Cosine similarity error:",
            e
        )

        return 0.0


# =========================================
# FIND CACHED ANSWER
# =========================================

def find_cached_answer(
    question,
    query_embedding,
    threshold=CACHE_SIMILARITY_THRESHOLD
):

    cache = load_cache()

    if not cache:

        print("CACHE EMPTY")

        return None

    best_match = None
    best_similarity = -1.0

    for item in cache:

        embedding = item.get(
            "embedding"
        )

        answer = item.get(
            "answer"
        )

        cached_question = item.get(
            "question"
        )

        if (
            not embedding
            or not answer
            or not cached_question
        ):
            continue

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_match = item

    print(
        f"Best cache similarity: "
        f"{best_similarity:.4f}"
    )

    # =====================================
    # CACHE HIT
    # =====================================

    if (
        best_match is not None
        and best_similarity >= threshold
    ):

        print("CACHE HIT")

        return {

            "question":
                best_match.get(
                    "question",
                    ""
                ),

            "answer":
                best_match.get(
                    "answer",
                    ""
                ),

            "similarity":
                best_similarity
        }

    # =====================================
    # CACHE MISS
    # =====================================

    print("CACHE MISS")

    return None


# =========================================
# STORE VERIFIED ANSWER
# =========================================

def store_verified_answer(
    question,
    answer,
    embedding
):

    if not question or not answer:

        print(
            "Cache store skipped: "
            "question or answer empty."
        )

        return

    if embedding is None:

        print(
            "Cache store skipped: "
            "embedding is empty."
        )

        return

    # =====================================
    # CONVERT EMBEDDING TO NORMAL LIST
    # =====================================

    try:

        embedding = np.asarray(
            embedding,
            dtype="float32"
        ).flatten().tolist()

    except Exception as e:

        print(
            "Embedding conversion error:",
            e
        )

        return

    # =====================================
    # LOAD EXISTING CACHE
    # =====================================

    cache = load_cache()

    # =====================================
    # EXACT DUPLICATE QUESTION
    # =====================================

    for item in cache:

        existing_question = item.get(
            "question",
            ""
        )

        if (
            existing_question.strip().lower()
            ==
            question.strip().lower()
        ):

            # Keep question exactly as provided
            item["question"] = question

            # Keep answer exactly as generated
            item["answer"] = answer

            # Replace only embedding
            item["embedding"] = embedding

            save_cache(cache)

            print(
                "Existing cache entry updated."
            )

            return

    # =====================================
    # NEW CACHE ENTRY
    # =====================================

    new_entry = {

        "question": question,

        "answer": answer,

        "embedding": embedding

    }

    cache.append(
        new_entry
    )

    save_cache(cache)

    print(
        f"Verified answer stored. "
        f"Total cached answers: {len(cache)}"
    )