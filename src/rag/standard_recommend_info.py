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

VECTOR_DB_INFO = VECTOR_DB_DIR / "vector_db_info"
COLLECTION_NAME = "standards_scope"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
OVERFETCH_K = 20

# =========================================================
# UTILITIES
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
# INPUT JSON → EMBEDDING TEXT
# (SCOPE + TESTS + REGION, NO HARD-CODED LOGIC)
# =========================================================

def input_json_to_embedding_text(input_json: dict) -> str:
    """
    Convert user input JSON into scope- and test-aligned
    plain text suitable for direct embedding.
    """

    parts = []

    product = input_json.get("product_details", {})
    req = input_json.get("testing_requirements", {})
    std = input_json.get("testing_standards", {})

    # ---- Scope / applicability intent ----
    if product.get("eut_name"):
        parts.append(product["eut_name"])

    if product.get("industry"):
        parts.append(product["industry"])

    if product.get("industry_other"):
        parts.append(product["industry_other"])

    if product.get("signal_lines"):
        parts.append(product["signal_lines"])

    # ---- Region (applicability context) ----
    if std.get("regions"):
        parts.extend(std["regions"])

    # ---- Tests (EXPLICIT – matches scope JSON test sections) ----
    if req.get("test_type"):
        parts.append(req["test_type"])

    if req.get("selected_tests"):
        parts.extend(req["selected_tests"])

    return "\n".join(parts).strip()

# =========================================================
# CORE RETRIEVAL LOGIC
# =========================================================

def retrieve_relevant_documents(embedding_text: str, top_k: int = TOP_K):
    if not embedding_text:
        return []

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_INFO),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        embedding_text,
        normalize_embeddings=True
    ).tolist()
    
    query_tokens = tokenize(embedding_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=OVERFETCH_K,
        include=["documents", "metadatas", "distances"]
    )

    similarities = [1.0 - d for d in results["distances"][0]]
    mean_sim = statistics.mean(similarities)
    std_sim = statistics.pstdev(similarities)

    ranked = []

    for i, similarity in enumerate(similarities):
        doc_text = results["documents"][0][i]
        doc_tokens = tokenize(doc_text)

        lexical = normalized_lexical_overlap(query_tokens, doc_tokens)

        score = z_score(similarity, mean_sim, std_sim) + lexical * 0.1

        ranked.append({
            "document_id": results["metadatas"][0][i]["document_id"],
            "document_text": doc_text,
            "similarity": round(similarity, 4),
            "score": round(score, 4)
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Standards Recommendation API",
    version="1.0"
)

class RecommendationRequest(BaseModel):
    input_json: dict
    top_k: int = TOP_K

class RecommendationResult(BaseModel):
    document_id: str
    document_text: str
    similarity: float
    score: float

@app.post("/recommend", response_model=List[RecommendationResult])
def recommend(req: RecommendationRequest):

    embedding_text = input_json_to_embedding_text(req.input_json)

    return retrieve_relevant_documents(
        embedding_text=embedding_text,
        top_k=req.top_k
    )