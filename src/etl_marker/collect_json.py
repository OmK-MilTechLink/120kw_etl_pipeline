import shutil
from fastapi import FastAPI

from src.path import MARKER_JSON_DIR, OUTPUT_JSON_DIR

app = FastAPI(title="Collect Marker JSON API")

# -------------------------
# Original logic
# -------------------------

def collect_marker_jsons():
    collected = []

    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    for doc_dir in MARKER_JSON_DIR.iterdir():
        if not doc_dir.is_dir():
            continue

        json_file = doc_dir / f"{doc_dir.name}.json"
        if json_file.exists():
            dest = OUTPUT_JSON_DIR / json_file.name
            shutil.copy2(json_file, dest)
            collected.append(json_file.name)

    return collected

# -------------------------
# API
# -------------------------

@app.post("/collect/json")
def collect_json():
    files = collect_marker_jsons()

    if not files:
        return {"status": "no_files", "count": 0}

    return {
        "status": "success",
        "count": len(files),
        "files": files,
        "output_dir": str(OUTPUT_JSON_DIR),
    }