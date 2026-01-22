import json
import re
from pathlib import Path
from typing import List, Dict, Set
from fastapi import FastAPI, HTTPException
from src.path import OUTPUT_SCHEMA_DIR, OUTPUT_DIR

# =========================================================
# Output directory
# =========================================================

CHUNK_DIR = OUTPUT_DIR / "output_json_chunk"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# INTERNAL REFERENCE PATTERNS (ROBUST, NON-HALLUCINATED)
# =========================================================

# Numeric-only references: 6, 6.1, 6.1.1, 10.2.3.4
NUMERIC_REF_PATTERN = re.compile(
    r'\b\d+(?:\.\d+){0,4}\b'
)

# Keyword-based internal references
# Clause 6.1, Section 2, Annex A, Subclause 7.3
KEYWORD_REF_PATTERN = re.compile(
    r'\b(Clause|Subclause|Section|Annex)\s+([A-Z]|\d+(?:\.\d+)*)',
    re.IGNORECASE
)

TABLE_PATTERN = re.compile(
    r'\bTable\s+\d+[A-Z]?\b',
    re.IGNORECASE
)

FIGURE_PATTERN = re.compile(
    r'\bFigure\s+\d+[A-Z]?\b',
    re.IGNORECASE
)

STANDARD_PATTERN = re.compile(
    r'\b(?:'
    r'BS\s+EN|'
    r'ISO\s*/\s*IEC|'
    r'ISO|IEC|IEEE|CISPR|EN|HD|IS|AIS|BIS|ITU[- ]T'
    r')'
    r'\s+'
    r'[A-Z]?\d+[A-Z0-9.\-]*'
    r'(?:\s*\([^)]+\))?'
    r'(?:\s*:\s*\d{4})?'
    r'\b',
    re.IGNORECASE
)

# =========================================================
# Helpers
# =========================================================

def safe_filename(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")

def extract_text_blocks(content: List[Dict]) -> List[str]:
    texts = []
    for item in content or []:
        t = item.get("text")
        if isinstance(t, str):
            texts.append(t)
    return texts

# =========================================================
# Reference Extraction
# =========================================================

def extract_internal_references(content: List[Dict]) -> Set[str]:
    texts = extract_text_blocks(content)
    refs = set()

    for text in texts:
        for kw, ref in KEYWORD_REF_PATTERN.findall(text):
            refs.add(f"{kw} {ref}")

        for num in NUMERIC_REF_PATTERN.findall(text):
            refs.add(num)

        for t in TABLE_PATTERN.findall(text):
            refs.add(t)

        for f in FIGURE_PATTERN.findall(text):
            refs.add(f)

    return refs

def extract_external_standards(content: List[Dict]) -> Set[str]:
    texts = extract_text_blocks(content)
    standards = set()

    for text in texts:
        for match in STANDARD_PATTERN.finditer(text):
            standards.add(match.group(0).strip())

    return standards

def resolve_internal_references(
    raw_refs: Set[str],
    known_chunk_ids: Set[str],
    document_id: str
) -> List[str]:
    """
    Resolve internal references ONLY if exact chunk ID exists.
    """
    resolved = set()

    for ref in raw_refs:
        ref_id = ref.split()[-1]  # handles "Clause 6.1" → "6.1"
        if ref_id in known_chunk_ids:
            resolved.add(f"{document_id}::{ref_id}")

    return sorted(resolved)

# =========================================================
# Chunk Writer
# =========================================================

def write_chunk(chunk: dict, known_chunk_ids: Set[str]):
    doc_id = chunk["document_id"]
    doc_dir = CHUNK_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract references ---
    internal_raw = extract_internal_references(chunk.get("content", []))
    external_standards = extract_external_standards(chunk.get("content", []))
    internal_resolved = resolve_internal_references(
        internal_raw,
        known_chunk_ids,
        doc_id
    )

    out_chunk = {
        "chunk_id": f"{doc_id}::{chunk['id']}",
        "document_id": doc_id,
        "clause_id": chunk["id"],
        "title": chunk.get("title"),
        "parent_id": chunk.get("parent_id"),
        "content": chunk.get("content", []),
        "tables": chunk.get("tables", []),
        "figures": chunk.get("figures", []),
        "requirements": chunk.get("requirements", []),
        "references": {
            "internal_raw": sorted(internal_raw),
            "internal_resolved": internal_resolved,
            "standards": sorted(external_standards)
        },
        "children_ids": chunk.get("children_ids", [])
    }

    out_path = doc_dir / f"{safe_filename(chunk['id'])}.json"
    out_path.write_text(
        json.dumps(out_chunk, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# =========================================================
# Main (UNCHANGED)
# =========================================================

def main():
    schema_files = list(OUTPUT_SCHEMA_DIR.glob("*_final_schema.json"))

    if not schema_files:
        print(f"No schema files found in {OUTPUT_SCHEMA_DIR}")
        return

    for schema_file in schema_files:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))

        doc_id = schema.get("document_id")
        chunks = schema.get("chunks", [])

        if not doc_id or not chunks:
            print(f"Skipping {schema_file.name}: invalid schema")
            continue

        # Build chunk ID index ONCE per document
        known_chunk_ids = {c["id"] for c in chunks if "id" in c}

        for chunk in chunks:
            write_chunk(chunk, known_chunk_ids)

        print(f"[OK] {doc_id} → {len(chunks)} chunks processed")

# =========================================================
# FASTAPI API (ONLY ADDITION)
# =========================================================

app = FastAPI(title="Schema to Chunks API")

@app.post("/chunks/build")
def build_chunks():
    schema_files = list(OUTPUT_SCHEMA_DIR.glob("*_final_schema.json"))
    if not schema_files:
        raise HTTPException(
            status_code=404,
            detail=f"No schema files found in {OUTPUT_SCHEMA_DIR}"
        )

    processed_docs = []

    for schema_file in schema_files:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))

        doc_id = schema.get("document_id")
        chunks = schema.get("chunks", [])

        if not doc_id or not chunks:
            continue

        known_chunk_ids = {c["id"] for c in chunks if "id" in c}
        for chunk in chunks:
            write_chunk(chunk, known_chunk_ids)

        processed_docs.append(doc_id)

    return {
        "status": "success",
        "documents_chunked": len(processed_docs),
        "documents": processed_docs
    }

if __name__ == "__main__":
    main()