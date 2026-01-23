import json
from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.path import OUTPUT_DIR, VECTOR_DB_DIR

# =========================================================
# CONFIG
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
VECTOR_DB_SCOPE = VECTOR_DB_DIR / "vector_db_scope"

COLLECTION_NAME = "standards_scope"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Scope + Tests Embedding API",
    version="2.0"
)

# =========================================================
# HELPER
# =========================================================

def build_embedding_text(data: dict) -> str:
    """
    Build embedding text using ONLY extracted content.
    No inference, no modification.
    """
    parts = []

    if data.get("document_name"):
        parts.append(f"DOCUMENT NAME:\n{data['document_name']}")

    if data.get("document_title"):
        parts.append(f"DOCUMENT TITLE:\n{data['document_title']}")

    scope = data.get("scope", [])
    if scope:
        parts.append("SCOPE:")
        parts.extend(scope)

    tests = data.get("tests", [])
    if tests:
        parts.append("TEST SECTIONS:")
        parts.extend(tests)

    return "\n\n".join(parts)

# =========================================================
# ENDPOINT
# =========================================================

@app.post("/embed")
def embed_all_scopes():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_SCOPE),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"level": "document_scope_and_tests"}
    )

    ids, documents, metadatas = [], [], []
    total = 0

    for scope_file in sorted(SCOPE_DIR.glob("*.json")):
        data = json.loads(scope_file.read_text(encoding="utf-8"))

        document_id = data.get("document_id")
        if not document_id:
            continue

        embedding_text = build_embedding_text(data)
        if not embedding_text:
            continue

        ids.append(document_id)
        documents.append(embedding_text)
        metadatas.append({"document_id": document_id})

        if len(documents) >= BATCH_SIZE:
            embeddings = model.encode(
                documents,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE
            ).tolist()

            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            total += len(ids)
            ids, documents, metadatas = [], [], []

    if documents:
        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total += len(ids)

    return {
        "status": "done",
        "documents_embedded": total,
        "collection": COLLECTION_NAME,
        "vector_db": str(VECTOR_DB_SCOPE)
    }