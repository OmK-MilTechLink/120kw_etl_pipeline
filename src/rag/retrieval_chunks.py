import re
from typing import Optional, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.path import VECTOR_DB_DIR

# =========================================================
# CONFIG
# =========================================================

VECTOR_DB_CHUNK = VECTOR_DB_DIR / "vector_db_chunk"
COLLECTION_NAME = "standards_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OVERFETCH_K = 50
TOP_K = 10

# =========================================================
# TEXT UTILITIES
# =========================================================

def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def normalized_lexical_overlap(query_tokens: set, text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)

def clause_depth(clause_id: str) -> int:
    return clause_id.count(".") if clause_id else 0

def parent_clause_id(clause_id: str):
    if "." not in clause_id:
        return None
    return clause_id.rsplit(".", 1)[0]

def is_direct_child(parent: str, child: str) -> bool:
    if not child.startswith(parent + "."):
        return False
    remainder = child[len(parent) + 1:]
    return "." not in remainder

# =========================================================
# CORE RETRIEVAL (FIXED, MINIMAL)
# =========================================================

def retrieve_best_chunks(query: str, top_k: int = TOP_K):

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_CHUNK),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    query_tokens = set(tokenize(query))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=OVERFETCH_K,
        include=["documents", "metadatas", "distances"]
    )

    base_chunks: Dict[str, dict] = {}

    # ---------------------------
    # Initial scoring (UNCHANGED)
    # ---------------------------
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        lexical = normalized_lexical_overlap(query_tokens, text)
        if lexical == 0.0:
            continue

        semantic = 1.0 - (distance/2.0) # since cosine similarity is between (2,0) and not (1,0)
        # so we need to half the distance calculated from the vectorDB to not get negative scores
        score = semantic * 0.75 + lexical * 0.25

        clause_id = meta.get("clause_id")
        if not clause_id:
            continue

        base_chunks[clause_id] = {
            "document_id": meta.get("document_id"),
            "clause_id": clause_id,
            "score": score,
            "semantic": semantic,
            "lexical": lexical,
            "depth": clause_depth(clause_id),
            "has_requirements": meta.get("has_requirements", False),
            "parent_clause": parent_clause_id(clause_id),
            "source": "direct"
        }

    # ---------------------------
    # SAFE, CORRECT EXPANSION
    # ---------------------------

    expanded = dict(base_chunks)

    # Only expand from top base hits
    top_base = sorted(
        base_chunks.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    for chunk in top_base:
        cid = chunk["clause_id"]

        # Add parent if already retrieved
        parent = chunk["parent_clause"]
        if parent and parent in base_chunks:
            if expanded[parent]["source"] == "direct":
                expanded[parent]["source"] = "parent"

        # Add direct children if already retrieved
        for other_id in base_chunks:
            if is_direct_child(cid, other_id):
                if expanded[other_id]["source"] == "direct":
                    expanded[other_id]["source"] = "child"

    # ---------------------------
    # FINAL RANKING + CONFIDENCE
    # ---------------------------

    ranked = list(expanded.values())
    ranked.sort(
        key=lambda x: (
            x["score"],
            x["has_requirements"]
        ),
        reverse=True
    )

    top = ranked[:top_k]

    max_score = top[0]["score"] if top else 1.0
    for r in top:
        r["confidence"] = round(r["score"] / max_score, 4)

    return top

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="Chunk Retrieval API", version="1.2")

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = TOP_K

class RetrievalResult(BaseModel):
    document_id: str
    clause_id: str
    score: float
    confidence: float
    semantic: float
    lexical: float
    depth: int
    has_requirements: bool
    parent_clause: Optional[str]
    source: str

@app.post("/retrieve", response_model=List[RetrievalResult])
def retrieve_chunks(req: RetrievalRequest):
    return retrieve_best_chunks(req.query, req.top_k)