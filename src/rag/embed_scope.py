import json
from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.path import OUTPUT_DIR, VECTOR_DB_DIR

# =========================================================
# CONFIG (UNCHANGED)
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
VECTOR_DB_SCOPE = VECTOR_DB_DIR / "vector_db_scope"

COLLECTION_NAME = "standards_scope"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64

# =========================================================
# FASTAPI APP (MINIMAL)
# =========================================================

app = FastAPI(
    title="Scope Embedding API",
    version="1.0"
)

# =========================================================
# SINGLE ENDPOINT
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
        metadata={"level": "document_scope"}
    )

    ids, documents, metadatas = [], [], []
    total = 0

    for scope_file in sorted(SCOPE_DIR.glob("*.json")):
        data = json.loads(scope_file.read_text(encoding="utf-8"))

        document_id = data.get("document_id")
        scope_lines = data.get("scope", [])

        if not document_id or not scope_lines:
            continue

        embedding_text = "\n\n".join(scope_lines)

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
