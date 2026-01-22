import subprocess
import shutil
import sys
from fastapi import FastAPI, HTTPException

from src.path import INPUT_PDFS_DIR, MARKER_JSON_DIR, COMPLETE_DIR

app = FastAPI(title="Marker Init API")

# -------------------------
# Helpers (original logic)
# -------------------------

def ensure_dirs():
    INPUT_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_JSON_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETE_DIR.mkdir(parents=True, exist_ok=True)

def get_input_pdfs():
    return list(INPUT_PDFS_DIR.glob("*.pdf"))

def move_to_completed(pdfs):
    for pdf in pdfs:
        shutil.move(str(pdf), str(COMPLETE_DIR / pdf.name))

def marker_cmd():
    return [
        "marker",
        str(INPUT_PDFS_DIR),
        "--output_dir",
        str(MARKER_JSON_DIR),
        "--workers",
        "1",
        "--output_format",
        "json",
    ]

# -------------------------
# API
# -------------------------

@app.post("/marker/run")
def run_marker():
    ensure_dirs()
    pdfs = get_input_pdfs()

    if not pdfs:
        return {"status": "no_input", "message": "No PDFs found"}

    try:
        subprocess.run(
            marker_cmd(),
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Marker failed")

    move_to_completed(pdfs)

    return {
        "status": "success",
        "processed_pdfs": len(pdfs),
        "completed_dir": str(COMPLETE_DIR),
    }