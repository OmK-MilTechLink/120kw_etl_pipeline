import json
import re
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any
from fastapi import FastAPI
from src.path import OUTPUT_DIR

# =========================================================
# CONFIG
# =========================================================

SOURCE_CHUNK_DIR = OUTPUT_DIR / "output_json_chunk"
SHORT_CHUNK_DIR = OUTPUT_DIR / "output_short_chunk"

MAX_CHARS = 2048
MAX_RECURSION_DEPTH = 20   # SAFETY NET ONLY

SHORT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Chunk Size Optimization API")

# =========================================================
# Regex
# =========================================================

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# =========================================================
# Length Calculation
# =========================================================

def get_all_text_length(obj: Any) -> int:
    """Recursively count all text in object."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(get_all_text_length(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(get_all_text_length(v) for v in obj)
    return 0

# =========================================================
# Text Splitting
# =========================================================

def split_text_hard(text: str, max_size: int) -> List[str]:
    if len(text) <= max_size:
        return [text]

    parts = []
    start = 0

    while start < len(text):
        end = start + max_size
        if end >= len(text):
            parts.append(text[start:])
            break

        segment = text[start:end]
        last_space = segment.rfind(" ")

        if last_space > max_size * 0.7:
            actual_end = start + last_space + 1
        else:
            actual_end = end

        parts.append(text[start:actual_end])
        start = actual_end

    return parts

def split_text_to_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s for s in sentences if s]

# =========================================================
# Block Splitting
# =========================================================

def split_text_blocks(blocks: List[Dict[str, Any]], max_chars: int):
    if not blocks:
        return []

    chunks = []
    current = []
    current_len = 0

    for block in blocks:
        text = block.get("text", "")
        if not isinstance(text, str) or not text:
            current.append(block)
            continue

        if len(text) <= max_chars:
            if current_len + len(text) > max_chars:
                chunks.append(current)
                current, current_len = [], 0

            current.append(block)
            current_len += len(text)
            continue

        # Split oversized block
        if current:
            chunks.append(current)
            current, current_len = [], 0

        sentences = split_text_to_sentences(text)
        if len(sentences) > 1:
            for sent in sentences:
                for frag in split_text_hard(sent, max_chars):
                    if current_len + len(frag) > max_chars:
                        chunks.append(current)
                        current, current_len = [], 0

                    new_block = deepcopy(block)
                    new_block["text"] = frag
                    current.append(new_block)
                    current_len += len(frag)
        else:
            for frag in split_text_hard(text, max_chars):
                if current_len + len(frag) > max_chars:
                    chunks.append(current)
                    current, current_len = [], 0

                new_block = deepcopy(block)
                new_block["text"] = frag
                current.append(new_block)
                current_len += len(frag)

    if current:
        chunks.append(current)

    return chunks

# =========================================================
# SAFE RECURSIVE SPLITTER
# =========================================================

def split_chunk_recursive(chunk: Dict[str, Any], depth: int = 0) -> List[Dict[str, Any]]:
    original_len = get_all_text_length(chunk)

    # Base cases
    if original_len <= MAX_CHARS:
        return [chunk]

    if depth >= MAX_RECURSION_DEPTH:
        return [chunk]

    # Only fields that can shrink
    splittable_fields = [
        field for field in ("content", "requirements")
        if field in chunk and isinstance(chunk[field], list) and chunk[field]
    ]

    if not splittable_fields:
        return [chunk]

    results = []

    for field in splittable_fields:
        parts = split_text_blocks(chunk[field], MAX_CHARS)

        for part in parts:
            new_chunk = deepcopy(chunk)

            for f in splittable_fields:
                new_chunk[f] = []

            new_chunk[field] = part
            new_len = get_all_text_length(new_chunk)

            # Progress guard
            if new_len >= original_len:
                results.append(new_chunk)
                continue

            if new_len <= MAX_CHARS:
                results.append(new_chunk)
            else:
                results.extend(
                    split_chunk_recursive(new_chunk, depth + 1)
                )

    return results

# =========================================================
# FILE PROCESSING
# =========================================================

def process_chunk_file(chunk_file: Path, output_dir: Path) -> int:
    chunk = json.loads(chunk_file.read_text(encoding="utf-8"))
    total_len = get_all_text_length(chunk)

    if total_len <= MAX_CHARS:
        out_path = output_dir / chunk_file.name
        out_path.write_text(
            json.dumps(chunk, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return 1

    sub_chunks = split_chunk_recursive(chunk)
    total_parts = len(sub_chunks)

    base_clause = chunk.get("clause_id", "unknown")
    doc_id = chunk.get("document_id", "unknown")

    for idx, sub in enumerate(sub_chunks, 1):
        new_clause_id = f"{base_clause}__part_{idx}_of_{total_parts}"

        sub["clause_id"] = new_clause_id
        sub["chunk_id"] = f"{doc_id}::{new_clause_id}"
        sub["chunk_part"] = idx
        sub["chunk_part_total"] = total_parts
        sub["original_clause_id"] = base_clause

        out_file = output_dir / f"{new_clause_id}.json"
        out_file.write_text(
            json.dumps(sub, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return total_parts

# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/split")
def split_all_chunks():
    original_count = 0
    output_count = 0

    for doc_dir in sorted(SOURCE_CHUNK_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue

        out_dir = SHORT_CHUNK_DIR / doc_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for file in sorted(doc_dir.glob("*.json")):
            original_count += 1
            output_count += process_chunk_file(file, out_dir)

    return {
        "status": "success",
        "original_chunks": original_count,
        "output_chunks": output_count,
        "output_directory": str(SHORT_CHUNK_DIR)
    }
