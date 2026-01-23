import re
import statistics
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.path import VECTOR_DB_DIR

# =========================================================
# CONFIG
# =========================================================

VECTOR_DB_SCOPE = VECTOR_DB_DIR / "vector_db_scope"
COLLECTION_NAME = "standards_scope"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
OVERFETCH_K = 20

# =========================================================
# UTILITIES (TEXT-AGNOSTIC)
# =========================================================

def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def normalized_lexical_overlap(query_tokens, text_tokens):
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)

def z_score(value, mean, std):
    if std == 0:
        return 0.0
    return (value - mean) / std

# =========================================================
# CORE RETRIEVAL
# =========================================================

def retrieve_relevant_documents(query: str, top_k: int = TOP_K):
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_SCOPE),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    query_tokens = tokenize(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=OVERFETCH_K,
        include=["documents", "metadatas", "distances"]
    )

    # Normalize cosine distance → similarity
    similarities = [1.0 - (d / 2.0) for d in results["distances"][0]]

    mean_sim = statistics.mean(similarities)
    std_sim = statistics.pstdev(similarities)

    ranked = []

    for i in range(len(results["ids"][0])):
        similarity = similarities[i]
        doc_text = results["documents"][0][i]
        doc_tokens = tokenize(doc_text)

        lexical = normalized_lexical_overlap(query_tokens, doc_tokens)

        final_score = z_score(similarity, mean_sim, std_sim) + lexical * 0.1

        ranked.append({
            "document_id": results["metadatas"][0][i]["document_id"],
            "similarity": round(similarity, 4),
            "score": round(final_score, 4)
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    cutoff = mean_sim - std_sim
    filtered = [r for r in ranked if r["similarity"] >= cutoff]

    if not filtered:
        filtered = ranked[:top_k]

    return filtered[:top_k]

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Standards Recommendation API",
    version="2.0"
)

# =========================================================
# REQUEST / RESPONSE
# =========================================================

class ScopeQuery(BaseModel):
    query: str
    top_k: int = TOP_K

class ScopeResult(BaseModel):
    document_id: str
    similarity: float
    score: float

# =========================================================
# ENDPOINT
# =========================================================

@app.post("/recommend", response_model=List[ScopeResult])
def recommend_standards(req: ScopeQuery):
    return retrieve_relevant_documents(
        query=req.query,
        top_k=req.top_k
    )