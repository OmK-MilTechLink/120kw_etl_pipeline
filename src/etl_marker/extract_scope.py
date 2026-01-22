import json
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException

from src.path import OUTPUT_JSON_DIR, OUTPUT_DIR

# =========================================================
# Output directory
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
SCOPE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Regex patterns (REAL, observed)
# =========================================================

SCOPE_HEADER_RE = re.compile(
    r">\s*(\d+\.?\s*)?Scope\s*<",
    re.IGNORECASE
)

CLAUSE_1_RE = re.compile(
    r">\s*1(\.|\s|<)",
    re.IGNORECASE
)

HTML_TAG_RE = re.compile(r"<[^>]+>")

# =========================================================
# Helpers
# =========================================================

def clean_html(html: str) -> str:
    return HTML_TAG_RE.sub("", html).strip()

# =========================================================
# Core extraction logic
# =========================================================

def extract_scope(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    scope_text = []
    collecting = False
    scope_hierarchy = None
    found_explicit_scope = False
    clause1_candidate = []

    for page in data.get("children", []):
        if page.get("block_type") != "Page":
            continue

        for block in page.get("children", []):
            block_type = block.get("block_type")
            html = block.get("html", "")
            hierarchy = block.get("section_hierarchy")

            # Explicit Scope detection
            if block_type in {"SectionHeader", "Text"} and SCOPE_HEADER_RE.search(html):
                collecting = True
                found_explicit_scope = True
                scope_hierarchy = hierarchy
                continue

            # Stop condition
            if collecting and block_type == "SectionHeader":
                if hierarchy and hierarchy != scope_hierarchy:
                    collecting = False

            # Collect Scope text
            if collecting and block_type == "Text":
                text = clean_html(html)
                if text:
                    scope_text.append(text)

            # Clause-1 fallback capture
            if not found_explicit_scope and block_type == "SectionHeader" and CLAUSE_1_RE.search(html):
                collecting = "clause1"
                continue

            if collecting == "clause1" and block_type == "Text":
                text = clean_html(html)
                if text:
                    clause1_candidate.append(text)

            if collecting == "clause1" and block_type == "SectionHeader":
                collecting = False

    # Final decision
    if scope_text:
        return scope_text

    if clause1_candidate:
        first = clause1_candidate[0].lower()
        if first.startswith(("this document", "this standard", "this part")):
            return clause1_candidate

    return []

# =========================================================
# FASTAPI API
# =========================================================

app = FastAPI(title="Scope Extraction API")

@app.post("/scope/extract")
def extract_scope_api():
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))

    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"No JSON files found in {OUTPUT_JSON_DIR}"
        )

    processed = []
    skipped = []

    for jf in json_files:
        scope = extract_scope(jf)

        if not scope:
            skipped.append(jf.name)
            continue

        out = {
            "document_id": jf.stem,
            "scope": scope
        }

        out_path = SCOPE_DIR / f"{jf.stem}_scope.json"
        out_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        processed.append(jf.stem)

    return {
        "status": "success",
        "documents_processed": len(processed),
        "documents_skipped": len(skipped),
        "processed_documents": processed,
        "skipped_documents": skipped,
        "output_dir": str(SCOPE_DIR)
    }

# =========================================================
# CLI compatibility
# =========================================================

if __name__ == "__main__":
    for jf in OUTPUT_JSON_DIR.glob("*.json"):
        scope = extract_scope(jf)
        if scope:
            out = {
                "document_id": jf.stem,
                "scope": scope
            }
            out_path = SCOPE_DIR / f"{jf.stem}_scope.json"
            out_path.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"[OK] Scope extracted → {out_path.name}")
