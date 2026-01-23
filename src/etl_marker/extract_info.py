"""
Standards Document Extraction Pipeline

Processes Marker-extracted JSON files from standards PDFs (IEC, ISO, IS/BIS, EN, IEEE).
Extracts document identity, title, scope, and ALL numbered test sections.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from fastapi import FastAPI, HTTPException

from src.path import OUTPUT_JSON_DIR, OUTPUT_DIR

# =========================================================
# Configuration
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
SCOPE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Regex Patterns
# =========================================================

# Scope section detection
SCOPE_HEADER_RE = re.compile(r">\s*(\d+\.?\s*)?(Scope|SCOPE)\s*<")
CLAUSE_1_RE = re.compile(r">\s*1(\.|\s|<)")

# Document ID patterns for different standards bodies
DOC_ID_PATTERNS = [
    # IEC: IEC 61076-8-103:2023
    r"\bIEC\s+\d+(?:[-/]\d+)*(?:\s*:\s*\d{4})?",
    # ISO: ISO 9001:2015, ISO/IEC 27001:2013
    r"\bISO(?:/IEC)?\s+\d+(?:[-/]\d+)*(?:\s*:\s*\d{4})?",
    # IS/BIS: IS 17017, IS 17017 Part 2 Section 2
    r"\bIS\s+\d+(?:\s+Part\s+\d+)?(?:\s+Sec(?:tion)?\s+\d+)?(?:\s*:\s*\d{4})?",
    # BS EN: BS EN 50174-3:2013
    r"\bBS\s+EN\s+\d+(?:[-/]\d+)*(?:\s*:\s*\d{4})?",
    # EN: EN 50174-3:2013
    r"\bEN\s+\d+(?:[-/]\d+)*(?:\s*:\s*\d{4})?",
    # IEEE: IEEE 802.11-2020
    r"\bIEEE\s+\d+(?:\.\d+)?(?:[-/]\d+)*(?:\s*:\s*\d{4})?",
]
DOC_ID_RE = re.compile("|".join(f"({p})" for p in DOC_ID_PATTERNS), re.IGNORECASE)

# HTML tag removal
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Test section detection - numbered sections with test/testing
TEST_SECTION_RE = re.compile(
    r"^\s*(\d+\.)*\d+\s+.*\btest(s|ing)?\b",
    re.IGNORECASE
)

# =========================================================
# Text Processing Utilities
# =========================================================

def clean_html(html: str) -> str:
    """Remove HTML tags, normalize whitespace, and fix encoding issues."""
    if not html:
        return ""
    
    text = HTML_TAG_RE.sub("", html)
    
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()

def is_english(text: str, threshold: float = 0.80) -> bool:
    """Check if text is primarily English based on ASCII character ratio."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > threshold

def contains_url_or_web_reference(text: str) -> bool:
    """Check if text contains URL patterns or web references."""
    web_patterns = [
        r"www\.",
        r"https?://",
        r"\.com\b",
        r"\.org\b",
        r"\.ch\b",
        r"\.net\b",
        r"webstore",
        r"search.*form",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in web_patterns)

def is_boilerplate_section(text: str) -> bool:
    """Check if text is a common boilerplate/administrative section."""
    boilerplate_keywords = [
        "foreword", "preface", "introduction",
        "copyright", "contents", "table of contents",
        "bibliography", "index", "references",
        "annex", "amendment", "corrigendum",
        "about the", "publication", "acknowledgement"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in boilerplate_keywords)

def starts_with_section_number(text: str) -> bool:
    """Check if text starts with a section number (e.g., '1.2', '3.4.5')."""
    return bool(re.match(r'^\d+(\.\d+)*\s+', text))

# =========================================================
# Document Name Extraction
# =========================================================

def extract_document_name(blocks: List[Dict]) -> Optional[str]:
    """
    Extract the most complete document standard number.
    Returns the longest matching pattern found.
    """
    candidates = []
    
    for block in blocks:
        text = clean_html(block.get("html", ""))
        if not text:
            continue
        
        match = DOC_ID_RE.search(text)
        if match:
            candidate = match.group(0).strip()
            candidates.append(candidate)
    
    # Return the longest (most complete) standard number
    return max(candidates, key=len) if candidates else None

# =========================================================
# Document Title Extraction
# =========================================================

def score_title_candidate(text: str, header_index: int) -> Tuple[bool, int]:
    """
    Score a potential title candidate.
    Returns (is_valid, score) where higher score = better candidate.
    """
    # Immediate disqualifiers
    if not text or not is_english(text):
        return (False, 0)
    
    if is_boilerplate_section(text):
        return (False, 0)
    
    if contains_url_or_web_reference(text):
        return (False, 0)
    
    if DOC_ID_RE.fullmatch(text):
        return (False, 0)
    
    if starts_with_section_number(text):
        return (False, 0)
    
    # Must appear reasonably early in document
    if header_index > 20:
        return (False, 0)
    
    # Calculate score based on characteristics
    word_count = len(text.split())
    
    # Must have minimum substance
    if word_count < 4:
        return (False, 0)
    
    # Scoring factors
    score = 0
    
    # Word count (longer titles are more descriptive)
    score += word_count * 10
    
    # Early position bonus (earlier = more likely to be title)
    position_bonus = max(0, 200 - (header_index * 10))
    score += position_bonus
    
    # Very long titles get a boost (substantive descriptions)
    if word_count >= 10:
        score += 50
    
    return (True, score)

def extract_document_title(blocks: List[Dict]) -> Optional[str]:
    """
    Extract the most likely document title from section headers.
    Uses scoring to select the best candidate.
    """
    candidates = []
    header_index = 0
    
    for block in blocks:
        if block.get("block_type") != "SectionHeader":
            continue
        
        text = clean_html(block.get("html", ""))
        if not text:
            continue
        
        is_valid, score = score_title_candidate(text, header_index)
        if is_valid:
            candidates.append((score, header_index, text))
        
        header_index += 1
    
    if not candidates:
        return None
    
    # Sort by score (highest first), then by position (earliest first)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]

# =========================================================
# Scope Extraction
# =========================================================

def extract_scope(json_path: Path) -> List[str]:
    """
    Extract scope section from Marker JSON.
    
    Strategy:
    1. Look for explicit "Scope" section header
    2. Collect Text blocks under same hierarchy
    3. Fallback: Use Clause 1 if it starts with standard document phrasing
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []

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

            # Detect explicit "Scope" section
            if block_type in {"SectionHeader", "Text"} and SCOPE_HEADER_RE.search(html):
                collecting = True
                found_explicit_scope = True
                scope_hierarchy = hierarchy
                continue

            # Stop collecting if hierarchy changes
            if collecting and block_type == "SectionHeader":
                if hierarchy and hierarchy != scope_hierarchy:
                    collecting = False

            # Collect scope text
            if collecting and block_type == "Text":
                text = clean_html(html)
                if text:
                    scope_text.append(text)

            # Fallback: Detect Clause 1
            if not found_explicit_scope and block_type == "SectionHeader" and CLAUSE_1_RE.search(html):
                collecting = "clause1"
                continue

            # Collect Clause 1 text
            if collecting == "clause1" and block_type == "Text":
                text = clean_html(html)
                if text:
                    clause1_candidate.append(text)

            # Stop Clause 1 collection at next header
            if collecting == "clause1" and block_type == "SectionHeader":
                collecting = False

    # Return explicit scope if found
    if scope_text:
        return scope_text

    # Return Clause 1 if it follows standard phrasing
    if clause1_candidate:
        first_para = clause1_candidate[0].lower()
        standard_phrasings = [
            "this document",
            "this standard",
            "this part",
            "this specification",
            "this section",
            "this international standard"
        ]
        if any(first_para.startswith(phrase) for phrase in standard_phrasings):
            return clause1_candidate

    return []

# =========================================================
# Test Section Extraction
# =========================================================

def extract_test_sections(blocks: List[Dict]) -> List[str]:
    """
    Extract ALL numbered test sections from blocks.
    Preserves clause numbers exactly as they appear.
    """
    tests = []
    seen: Set[str] = set()

    for block in blocks:
        if block.get("block_type") != "SectionHeader":
            continue

        text = clean_html(block.get("html", ""))
        
        # Include only numbered sections containing "test" or "testing"
        if not TEST_SECTION_RE.match(text):
            continue

        # Deduplicate while preserving order
        if text not in seen:
            tests.append(text)
            seen.add(text)

    return tests

# =========================================================
# Main Extraction Pipeline
# =========================================================

def extract_document_and_tests(json_path: Path) -> Dict:
    """Extract document identity, title, and ALL test sections."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {
            "document_name": json_path.stem,
            "document_title": json_path.stem,
            "tests": []
        }

    # Flatten all blocks from all pages
    blocks = []
    for page in data.get("children", []):
        if page.get("block_type") == "Page":
            blocks.extend(page.get("children", []))

    # Extract components
    document_name = extract_document_name(blocks)
    document_title = extract_document_title(blocks)
    tests = extract_test_sections(blocks)

    # Fallback to filename if needed
    if not document_name:
        document_name = json_path.stem

    if not document_title:
        document_title = document_name

    return {
        "document_name": document_name,
        "document_title": document_title,
        "tests": tests
    }

# =========================================================
# Document Processing
# =========================================================

def process_document(json_path: Path) -> Dict:
    """Process a single document and return complete metadata."""
    scope = extract_scope(json_path)
    meta = extract_document_and_tests(json_path)

    return {
        "document_id": json_path.stem,
        "document_name": meta["document_name"],
        "document_title": meta["document_title"],
        "scope": scope,
        "tests": meta["tests"]
    }

def save_document(output: Dict, output_dir: Path) -> Path:
    """Save document metadata to JSON file."""
    output_path = output_dir / f"{output['document_id']}_scope.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return output_path

# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(title="Standards Document Extraction API")

@app.post("/scope/extract")
def extract_scope_api():
    """Extract scope and tests from all JSON files."""
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))

    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"No JSON files found in {OUTPUT_JSON_DIR}"
        )

    processed = []
    skipped = []

    for json_file in json_files:
        output = process_document(json_file)

        if not output["scope"] and not output["tests"]:
            skipped.append(json_file.stem)
            continue

        save_document(output, SCOPE_DIR)
        processed.append(json_file.stem)

    return {
        "status": "success",
        "documents_processed": len(processed),
        "documents_skipped": len(skipped),
        "processed_documents": processed,
        "skipped_documents": skipped,
        "output_dir": str(SCOPE_DIR)
    }

# =========================================================
# CLI Execution
# =========================================================

if __name__ == "__main__":
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))

    for json_file in json_files:
        output = process_document(json_file)

        if output["scope"] or output["tests"]:
            save_document(output, SCOPE_DIR)
            print(f"[OK] {json_file.name} → {output['document_id']}_scope.json")
        else:
            print(f"[SKIP] {json_file.name}")