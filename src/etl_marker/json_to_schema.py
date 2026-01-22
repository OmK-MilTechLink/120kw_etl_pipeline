import json
import re
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from src.path import OUTPUT_JSON_DIR, OUTPUT_SCHEMA_DIR, OUTPUT_DIR

# =========================================================
# REGEX PATTERNS - For extracting structured information
# =========================================================

CLAUSE_WITH_TITLE_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s+(.+)$', re.IGNORECASE)
CLAUSE_NUM_ONLY_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s*$', re.IGNORECASE)
HTML_TAG_RE = re.compile(r'<[^>]+>')
REQ_RE = re.compile(r'\b(shall not|shall|should|may)\b', re.IGNORECASE)
TABLE_REF_RE = re.compile(r'\btable\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
FIGURE_REF_RE = re.compile(r'\b(?:figure|fig\.?)\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)

# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class Requirement:
    type: str
    keyword: str
    text: str

@dataclass
class ContentItem:
    type: str
    text: str

@dataclass
class TableEntry:
    html: str
    number: Optional[str] = None
    caption: Optional[str] = None
    rows: Optional[List[Any]] = None

@dataclass
class FigureEntry:
    number: Any
    path: str
    format: str
    original_key: str
    size_bytes: int
    caption: Optional[str] = None

@dataclass
class Clause:
    id: str
    title: str
    parent_id: Optional[str] = None
    children: List['Clause'] = field(default_factory=list)
    content: List[ContentItem] = field(default_factory=list)
    tables: List[TableEntry] = field(default_factory=list)
    figures: List[FigureEntry] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    references: Dict[str, List[str]] = field(default_factory=lambda: {"internal": [], "external": []})

    def to_dict(self, doc_id: str) -> Dict:
        return {
            "id": self.id,
            "document_id": doc_id,
            "title": self.title,
            "parent_id": self.parent_id,
            "content": [{"type": c.type, "text": c.text} for c in self.content],
            "tables": [self._table_to_dict(t) for t in self.tables],
            "figures": [self._figure_to_dict(f) for f in self.figures],
            "requirements": [{"type": r.type, "keyword": r.keyword, "text": r.text} for r in self.requirements],
            "references": self.references,
            "children_ids": [c.id for c in self.children],
        }

    @staticmethod
    def _table_to_dict(t: TableEntry) -> Dict:
        out = {"number": t.number, "caption": t.caption}
        if t.html:
            out["html"] = t.html
        return out

    @staticmethod
    def _figure_to_dict(f: FigureEntry) -> Dict:
        return {
            "number": f.number,
            "path": f.path,
            "format": f.format,
            "caption": f.caption,
            "size_bytes": f.size_bytes,
        }

def strip_html(html: str) -> str:
    """Remove HTML tags and clean whitespace from text."""
    if not html:
        return ""
    return HTML_TAG_RE.sub('', html).strip()

def detect_image_format(data: bytes) -> str:
    """Detect image format from binary data header and return file extension."""
    if not data:
        return ".bin"
    
    format_signatures = [
        (b"\x89PNG", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"GIF8", ".gif"),
        (b"BM", ".bmp"),
    ]
    
    for signature, ext in format_signatures:
        if data.startswith(signature):
            return ext
    
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    
    return ".bin"

def extract_clause_info(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract clause ID and title from text.
    Returns: (clause_id, title) or None
    """
    if not text:
        return None
    
    # Try matching clause with title
    match = CLAUSE_WITH_TITLE_RE.match(text)
    if match:
        clause_id = text.split(maxsplit=1)[0]
        title = text[len(clause_id):].strip()
        return (clause_id, title if title else None)
    
    # Try matching clause number only
    match = CLAUSE_NUM_ONLY_RE.match(text)
    if match:
        return (match.group(0).strip(), None)
    
    return None

def extract_table_number(caption: str) -> Optional[str]:
    """Extract table number from caption."""
    if not caption:
        return None
    match = TABLE_REF_RE.search(caption)
    return match.group(1) if match else None

def extract_figure_number(caption: str) -> Optional[str]:
    """Extract figure number from caption."""
    if not caption:
        return None
    match = FIGURE_REF_RE.search(caption)
    return match.group(1) if match else None

def extract_requirements(text: str) -> List[Requirement]:
    """Extract normative requirements from text based on keywords."""
    if not text:
        return []
    
    requirement_map = {
        "shall not": "prohibition",
        "shall": "mandatory",
        "should": "recommendation",
        "may": "permission"
    }
    
    requirements = []
    for match in REQ_RE.finditer(text):
        keyword = match.group(1).lower()
        req_type = requirement_map.get(keyword)
        
        if req_type:
            requirements.append(Requirement(
                type=req_type,
                keyword=keyword,
                text=text
            ))
            break  # Only capture first requirement per text block
    
    return requirements

def extract_references(text: str) -> List[str]:
    """
    Extract references to clauses, tables, and figures from text.
    Examples: "see clause 1.2", "Table 3.1", "Figure A.2"
    """
    if not text:
        return []
    
    references = []
    
    # Match clause references: "clause 1.2", "section A.3", etc.
    clause_pattern = re.compile(r'\b(?:clause|section|paragraph)\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
    for match in clause_pattern.finditer(text):
        references.append(f"clause:{match.group(1)}")
    
    # Match table references: "Table 1.2", etc.
    table_pattern = re.compile(r'\btable\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
    for match in table_pattern.finditer(text):
        references.append(f"table:{match.group(1)}")
    
    # Match figure references: "Figure 3.1", "Fig. 2", etc.
    figure_pattern = re.compile(r'\b(?:figure|fig\.?)\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
    for match in figure_pattern.finditer(text):
        references.append(f"figure:{match.group(1)}")
    
    return list(set(references))  # Remove duplicates

# =========================================================
# PROCESSING CONTEXT - Tracks state during document parsing
# =========================================================

@dataclass
class ProcessingContext:
    """Maintains state while processing the document tree."""
    current_clause_id: Optional[str] = None
    pending_number: Optional[str] = None
    pending_caption: Optional[str] = None
    
    def reset_pending(self):
        """Clear pending items when starting new clause."""
        self.pending_caption = None

@dataclass
class ProcessingCounters:
    """Track statistics during processing."""
    total_images: int = 0
    clause_images: int = 0
    misc_images: int = 0
    misc_image_counter: int = 0
    total_tables: int = 0
    figure_counters: Dict[str, int] = field(default_factory=dict)
    misc_image_metadata: List[Dict] = field(default_factory=list)

# =========================================================
# BLOCK PROCESSORS - Handlers for different block types
# =========================================================

class BlockProcessor:
    """Handles processing of different block types."""
    
    def __init__(self, clauses: Dict[str, Clause], context: ProcessingContext, 
                 counters: ProcessingCounters, img_root: Path, misc_img_dir: Path):
        self.clauses = clauses
        self.context = context
        self.counters = counters
        self.img_root = img_root
        self.misc_img_dir = misc_img_dir
    
    def process_section_header(self, block: Dict):
        """Process section header blocks."""
        text = strip_html(block.get("html", ""))
        
        if not text:
            return
        
        info = extract_clause_info(text)
        
        if info:
            clause_id, title = info
            
            if title:
                self._create_clause(clause_id, title)
            else:
                self.context.pending_number = clause_id
        
        elif self.context.pending_number:
            self._create_clause(self.context.pending_number, text)
    
    def _create_clause(self, clause_id: str, title: str):
        """Create a new clause."""
        if clause_id not in self.clauses:
            self.clauses[clause_id] = Clause(id=clause_id, title=title)
        self.context.current_clause_id = clause_id
        self.context.pending_number = None
        self.context.reset_pending()
    
    def process_caption(self, block: Dict):
        """Process caption blocks."""
        text = strip_html(block.get("html", ""))
        if not text:
            return
        
        if extract_table_number(text) or extract_figure_number(text):
            self.context.pending_caption = text
        else:
            self._add_to_current_clause_content(ContentItem("caption", text))
    
    def process_table(self, block: Dict):
        """Process table blocks."""
        html = block.get("html", "")
        if not html:
            return
        
        self.counters.total_tables += 1
        
        caption = self.context.pending_caption or strip_html(block.get("caption", ""))
        table_number = extract_table_number(caption) if caption else None
        
        table = TableEntry(
            html=html,
            number=table_number,
            caption=caption if caption else None,
            rows=block.get("rows")
        )
        
        if self.context.current_clause_id and self.context.current_clause_id in self.clauses:
            self.clauses[self.context.current_clause_id].tables.append(table)
        
        self.context.pending_caption = None
    
    def process_picture(self, block: Dict):
        """Process picture blocks."""
        images = block.get("images", {})
        
        if not images:
            return
        
        for img_key, b64_data in images.items():
            self._process_single_image(img_key, b64_data, block)
        
        self.context.pending_caption = None
    
    def _process_single_image(self, img_key: str, b64_data: str, block: Dict):
        """Process a single image."""
        try:
            if not b64_data:
                print(f"    Warning: Empty image data for {img_key}")
                return
            
            data = base64.b64decode(b64_data)
            ext = detect_image_format(data)
            self.counters.total_images += 1
            
            caption = self.context.pending_caption or strip_html(block.get("caption", ""))
            figure_number = extract_figure_number(caption) if caption else None
            
            img_ref = img_key.replace("/", "_").strip("_").lower()
            
            if self.context.current_clause_id and self.context.current_clause_id in self.clauses:
                self._save_clause_image(img_key, data, ext, img_ref, caption, figure_number)
            else:
                self._save_misc_image(img_key, data, ext, img_ref, caption)
        
        except Exception as e:
            print(f"    Warning: Failed to process image {img_key}: {e}")
    
    def _save_clause_image(self, img_key: str, data: bytes, ext: str, 
                          img_ref: str, caption: Optional[str], figure_number: Optional[str]):
        """Save image to clause folder."""
        cid = self.context.current_clause_id
        
        if cid not in self.counters.figure_counters:
            self.counters.figure_counters[cid] = 0
        self.counters.figure_counters[cid] += 1
        
        clause_dir = self.img_root / cid.replace(".", "_")
        clause_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"figure_{self.counters.figure_counters[cid]}_{img_ref}{ext}"
        fpath = clause_dir / fname
        fpath.write_bytes(data)
        
        figure = FigureEntry(
            number=figure_number or self.counters.figure_counters[cid],
            path=str(fpath.relative_to(OUTPUT_DIR)),
            format=ext.lstrip("."),
            original_key=img_key,
            size_bytes=len(data),
            caption=caption
        )
        
        self.clauses[self.context.current_clause_id].figures.append(figure)
        self.counters.clause_images += 1
    
    def _save_misc_image(self, img_key: str, data: bytes, ext: str, 
                        img_ref: str, caption: Optional[str]):
        """Save image to misc folder."""
        self.counters.misc_image_counter += 1
        fname = f"misc_{self.counters.misc_image_counter}_{img_ref}{ext}"
        fpath = self.misc_img_dir / fname
        fpath.write_bytes(data)
        self.counters.misc_images += 1
        
        self.counters.misc_image_metadata.append({
            "path": str(fpath.relative_to(OUTPUT_DIR)),
            "caption": caption,
            "format": ext.lstrip("."),
            "original_key": img_key,
            "size_bytes": len(data)
        })
    
    def process_text(self, block: Dict):
        """Process text blocks."""
        text = strip_html(block.get("html", ""))
        
        if text and self._has_current_clause():
            clause = self.clauses[self.context.current_clause_id]
            clause.content.append(ContentItem("paragraph", text))
            clause.requirements.extend(extract_requirements(text))
            
            # Extract and store references
            refs = extract_references(text)
            for ref in refs:
                if ref.startswith("clause:"):
                    clause.references["internal"].append(ref.replace("clause:", ""))
                elif ref.startswith("table:") or ref.startswith("figure:"):
                    clause.references["internal"].append(ref)
    
    def process_footnote(self, block: Dict):
        """Process footnote blocks."""
        text = strip_html(block.get("html", ""))
        if text:
            self._add_to_current_clause_content(ContentItem("footnote", text))
    
    def process_list_item(self, block: Dict):
        """Process list item blocks."""
        text = strip_html(block.get("html", ""))
        
        if text and self._has_current_clause():
            clause = self.clauses[self.context.current_clause_id]
            clause.content.append(ContentItem("list_item", text))
            clause.requirements.extend(extract_requirements(text))
    
    def _has_current_clause(self) -> bool:
        """Check if there's a current clause context."""
        return self.context.current_clause_id and self.context.current_clause_id in self.clauses
    
    def _add_to_current_clause_content(self, item: ContentItem):
        """Add content item to current clause."""
        if self._has_current_clause():
            self.clauses[self.context.current_clause_id].content.append(item)

# =========================================================
# HIERARCHY BUILDING
# =========================================================

def build_clause_hierarchy(clauses: Dict[str, Clause]) -> List[Clause]:
    """Build hierarchical clause structure with parent_id relationships."""
    # Link children to parents and set parent_id
    for cid, clause in clauses.items():
        if cid[0].isdigit():
            # Numeric clause: 1.1.2 -> parent is 1.1
            parts = cid.split(".")
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                if parent_id in clauses:
                    parent = clauses[parent_id]
                    if clause not in parent.children:
                        parent.children.append(clause)
                        clause.parent_id = parent_id
        
        elif "." in cid:
            # Annex sub-clause: A.1 -> parent is A
            parts = cid.split(".")
            parent_id = parts[0]
            if parent_id in clauses:
                parent = clauses[parent_id]
                if clause not in parent.children:
                    parent.children.append(clause)
                    clause.parent_id = parent_id
    
    # Find root clauses (no parent)
    child_ids = {c.id for cl in clauses.values() for c in cl.children}
    roots = [c for cid, c in clauses.items() if cid not in child_ids]
    
    # Sort: numeric first, then alphabetic
    def sort_key(clause):
        cid = clause.id
        if cid[0].isdigit():
            try:
                return (0, [int(n) for n in cid.split(".")])
            except ValueError:
                return (0, [0])
        else:
            return (1, cid)
    
    roots.sort(key=sort_key)
    return roots

def flatten_clauses(clauses: List[Clause], doc_id: str) -> List[Dict]:
    """
    Flatten the hierarchical clause structure into a list of chunks.
    Each chunk contains only IDs for children, not nested objects.
    """
    chunks = []
    
    def process_clause(clause: Clause):
        chunks.append(clause.to_dict(doc_id))
        for child in clause.children:
            process_clause(child)
    
    for clause in clauses:
        process_clause(clause)
    
    return chunks

# =========================================================
# MAIN PROCESSING LOGIC
# =========================================================

def process_block(block: Dict, processor: BlockProcessor):
    """Recursively process a block from Marker JSON."""
    if not isinstance(block, dict):
        return
    
    btype = block.get("block_type")
    
    # Skip headers/footers
    if btype in ("PageHeader", "PageFooter"):
        return
    
    # Dispatch to appropriate handler
    handlers = {
        "SectionHeader": processor.process_section_header,
        "Caption": processor.process_caption,
        "Table": processor.process_table,
        "Picture": processor.process_picture,
        "Text": processor.process_text,
        "Footnote": processor.process_footnote,
        "ListItem": processor.process_list_item
    }
    
    handler = handlers.get(btype)
    if handler:
        handler(block)
    
    # Process children recursively
    children = block.get("children")
    if children and isinstance(children, list):
        for child in children:
            process_block(child, processor)

# =========================================================
# FILE CONVERSION - Main entry point
# =========================================================

def convert_file(path: Path) -> Dict:
    """Convert a Marker JSON file to structured schema."""
    # Load source JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    
    # Setup output directories
    doc_id = path.stem
    img_root = OUTPUT_DIR / "output_images" / doc_id
    img_root.mkdir(parents=True, exist_ok=True)
    
    misc_img_dir = img_root / "misc"
    misc_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data structures
    clauses: Dict[str, Clause] = OrderedDict()
    context = ProcessingContext()
    counters = ProcessingCounters()
    
    # Create processor
    processor = BlockProcessor(clauses, context, counters, img_root, misc_img_dir)
    
    # Process all blocks
    children = raw.get("children", [])
    for child in children:
        process_block(child, processor)
    
    # Build hierarchy
    roots = build_clause_hierarchy(clauses)
    
    # Flatten to chunks (each clause becomes a separate chunk with children_ids)
    chunks = flatten_clauses(roots, doc_id)
    
    # Build final result
    result = {
        "document_id": doc_id,
        "statistics": {
            "total_images": counters.total_images,
            "images_in_clauses": counters.clause_images,
            "images_in_misc": counters.misc_images,
            "total_tables": counters.total_tables,
            "total_clauses": len(clauses),
            "total_chunks": len(chunks)
        },
        "chunks": chunks
    }
    
    # Add misc images/tables as separate nodes if any
    if counters.misc_images > 0:
        misc_node = {
            "id": f"{doc_id}_misc",
            "document_id": doc_id,
            "title": "Miscellaneous Images",
            "parent_id": None,
            "content": [],
            "tables": [],
            "figures": counters.misc_image_metadata,
            "requirements": [],
            "references": {"internal": [], "external": []},
            "children_ids": []
        }
        result["chunks"].append(misc_node)
        result["statistics"]["total_chunks"] += 1
    
    return result

# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    """Process all JSON files in input directory."""
    OUTPUT_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {OUTPUT_JSON_DIR}")
        return
    
    print(f"Found {len(json_files)} file(s) to process\n")
    
    for file in json_files:
        print(f"Processing: {file.name}")
        
        try:
            schema = convert_file(file)
            
            out_path = OUTPUT_SCHEMA_DIR / f"{file.stem}_final_schema.json"
            out_path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            
            stats = schema['statistics']
            print(f"  [OK] Extracted {stats['total_images']} images")
            print(f"    - {stats['images_in_clauses']} in clauses")
            print(f"    - {stats['images_in_misc']} in misc folder")
            print(f"  [OK] Extracted {stats['total_tables']} tables")
            print(f"  [OK] Processed {stats['total_clauses']} clauses")
            print(f"  [OK] Generated {stats['total_chunks']} chunks")
            print(f"  [OK] Saved to: {out_path.name}\n")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            print()

# =========================================================
# FASTAPI API (ADDITION ONLY)
# =========================================================

app = FastAPI(title="JSON to Schema API")

@app.post("/schema/build")
def build_schema():
    OUTPUT_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="No JSON files found")

    processed = []

    for file in json_files:
        schema = convert_file(file)
        out_path = OUTPUT_SCHEMA_DIR / f"{file.stem}_final_schema.json"
        out_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        processed.append(out_path.name)

    return {
        "status": "success",
        "schemas_created": len(processed),
        "files": processed,
    }
