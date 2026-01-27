from pathlib import Path
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

from src.path import OUTPUT_DIR, VECTOR_DB_DIR

# =========================
# CONFIG (UNCHANGED)
# =========================

CHUNKS_ROOT = OUTPUT_DIR / "output_short_chunk"
VECTOR_DB_CHUNK = VECTOR_DB_DIR / "vector_db_chunk"
COLLECTION_NAME = "standards_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100

# =========================
# TEXT BUILDERS (UNCHANGED)
# =========================

def join_content_blocks(content):
    texts = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            texts.append(block["text"])
    return "\n".join(texts)


def join_requirements(requirements):
    texts = []
    for req in requirements:
        if isinstance(req, dict) and "text" in req:
            texts.append(req["text"])
    return "\n".join(texts)


def build_embedding_text(chunk: dict) -> str:
    sections = []

    sections.append(f"Document ID: {chunk.get('document_id', '')}")
    sections.append(f"Title: {chunk.get('title', '')}")
    sections.append(f"Clause ID: {chunk.get('clause_id', '')}")

    parent_id = chunk.get("parent_id")
    if parent_id:
        sections.append(f"Parent Clause: {parent_id}")

    content_text = join_content_blocks(chunk.get("content", []))
    if content_text:
        sections.append("CONTENT:")
        sections.append(content_text)

    req_text = join_requirements(chunk.get("requirements", []))
    if req_text:
        sections.append("REQUIREMENTS:")
        sections.append(req_text)

    refs = chunk.get("references", {})
    if isinstance(refs, dict):
        if refs.get("internal_raw"):
            sections.append("INTERNAL REFERENCES:")
            sections.append("\n".join(refs["internal_raw"]))

        if refs.get("standards"):
            sections.append("EXTERNAL STANDARDS:")
            sections.append("\n".join(refs["standards"]))

    for table in chunk.get("tables", []):
        html = table.get("html")
        if html:
            sections.append("TABLE:")
            sections.append(html)

    for fig in chunk.get("figures", []):
        caption = fig.get("caption")
        if caption:
            sections.append("FIGURE:")
            sections.append(caption)

    return "\n\n".join(sections)


def build_metadata(chunk: dict) -> dict:
    metadata = {
        "document_id": chunk.get("document_id", ""),
        "clause_id": chunk.get("clause_id", ""),
        "title": (chunk.get("title") or "")[:500],
        "has_requirements": bool(chunk.get("requirements")),
        "has_tables": bool(chunk.get("tables")),
        "has_figures": bool(chunk.get("figures")),
    }

    if chunk.get("parent_id"):
        metadata["parent_id"] = str(chunk["parent_id"])

    if chunk.get("chunk_part") is not None:
        metadata["chunk_part"] = int(chunk["chunk_part"])

    if chunk.get("chunk_part_total") is not None:
        metadata["chunk_part_total"] = int(chunk["chunk_part_total"])

    return metadata

# =========================
# FASTAPI APP (MINIMAL)
# =========================

app = FastAPI(
    title="Chunk Embedding API",
    version="1.5"
)

# =========================
# SINGLE ENDPOINT
# =========================

@app.post("/embed")
def embed_all_chunks():
    model = SentenceTransformer(MODEL_NAME)

    VECTOR_DB_CHUNK.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_CHUNK),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"domain": "electronics_standards"}
    )

    texts, metadatas, ids = [], [], []
    total_processed = 0

    for standard_dir in sorted(CHUNKS_ROOT.iterdir()):
        if not standard_dir.is_dir():
            continue

        for json_file in sorted(standard_dir.glob("*.json")):
            chunk = json.loads(json_file.read_text(encoding="utf-8"))

            texts.append(build_embedding_text(chunk))
            metadatas.append(build_metadata(chunk))
            ids.append(chunk.get("chunk_id", json_file.stem))
            total_processed += 1

            if len(texts) >= BATCH_SIZE:
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True
                ).tolist()

                collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                texts, metadatas, ids = [], [], []

    if texts:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return {
        "status": "done",
        "total_chunks_embedded": total_processed,
        "collection": COLLECTION_NAME,
        "vector_db": str(VECTOR_DB_CHUNK)
    }