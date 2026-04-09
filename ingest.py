# ingest.py (patched)

"""
Ingest content files, chunk them, embed them, and build a FAISS index.

Usage (locally):
    python ingest.py

On Render:
    Make sure this runs in the build command so the index exists
    before the app starts.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

from scribe.text_seam import extract_chunk_extractions


# ---------------------------
# Paths & constants
# ---------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # ensure your files are here

# Index directory:
# - Locally: defaults to <repo>/index
# - On Render (or other hosts): set INDEX_DIR env var, e.g. /var/data/index
DEFAULT_INDEX_DIR = ROOT / "index"
INDEX_DIR = Path(os.getenv("INDEX_DIR") or DEFAULT_INDEX_DIR)

INDEX_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONFIG_PATH = INDEX_DIR / "config.json"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "index.faiss"
FAISS_PATH_LEGACY = INDEX_DIR / "faiss.index"   # <— legacy filename some envs still point to

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


# ---------------------------
# Utilities
# ---------------------------

def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader  # lazy import
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in [".docx"]:
        from docx import Document  # type: ignore
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def iter_source_files() -> List[Path]:
    exts = {".txt", ".md", ".pdf", ".docx"}
    files: List[Path] = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def simple_chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
    """Naive character-based chunking with overlap."""

    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            # Heuristic priority: glossaries, rank, technique descriptions get higher
            lower_source = source.lower()
            if "glossary" in lower_source:
                priority = 3
            elif "rank" in lower_source:
                priority = 3
            elif "technique description" in lower_source or "technique_descriptions" in lower_source:
                priority = 3
            elif "kihon" in lower_source or "sanshin" in lower_source:
                priority = 2
            else:
                priority = 1
            extractions = extract_chunk_extractions(chunk_text, source)

            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "meta": {
                        "priority": priority,
                        # IMPORTANT: app.py's retrieve() expects meta["source"] for filename heuristics
                        "source": source,
                        "source_kind": extractions["source_kind"],
                        "extractions": extractions,
                    },
                }
            )

        if end == n:
            break
        start = end - CHUNK_OVERLAP

    return chunks


# ---------------------------
# Embeddings & index build
# ---------------------------

def embed_chunks(model: Any, chunks: List[Dict[str, Any]]) -> Any:
    if np is None:
        raise RuntimeError(
            "numpy is not installed. Install requirements.txt before rebuilding the index."
        )
    # Build embeddings from the exact list we will later pickle.
    texts = [(c.get("text") or "") for c in chunks]
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # preferred; keeps behavior consistent for IndexFlatIP
    )
    emb = np.asarray(emb, dtype="float32")
    return emb


def build_faiss_index(embeddings: Any) -> Any:
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. Install requirements.txt before rebuilding the index."
        )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # embeddings are already normalized above; keep this harmless as a belt-and-suspenders
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main() -> None:
    if np is None:
        raise RuntimeError(
            "numpy is not installed. Install requirements.txt before running ingest.py."
        )
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. Install requirements.txt before running ingest.py."
        )
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements.txt before running ingest.py."
        )

    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")

    files = iter_source_files()
    if not files:
        raise RuntimeError(f"No source files found in {DATA_DIR}")

    print("Found source files:")
    for f in files:
        print(" -", f.relative_to(ROOT))

    all_chunks: List[Dict[str, Any]] = []
    for f in files:
        print(f"\nReading {f} ...")
        text = read_text_file(f)
        print(f"  Length: {len(text)} characters")

        cks = simple_chunk_text(text, source=str(f.relative_to(ROOT)))
        print(f"  -> {len(cks)} chunks")
        all_chunks.extend(cks)

    print(f"\nTotal chunks (pre-filter): {len(all_chunks)}")

    # --- HARDENING: ensure we don't embed/persist empty chunks, and keep 1:1 alignment
    filtered_chunks: List[Dict[str, Any]] = []
    dropped_empty = 0
    for c in all_chunks:
        txt = (c.get("text") or "").strip()
        if not txt:
            dropped_empty += 1
            continue
        filtered_chunks.append(c)

    if dropped_empty:
        print(f"Dropped {dropped_empty} empty chunks")

    all_chunks = filtered_chunks
    print(f"Total chunks (post-filter): {len(all_chunks)}")

    print("\nLoading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks...")
    emb = embed_chunks(model, all_chunks)
    print("Embeddings shape:", emb.shape)

    # --- HARDENING: assert alignment before index build
    if emb.shape[0] != len(all_chunks):
        raise RuntimeError(
            f"BUG: embeddings/chunks mismatch before index build: "
            f"embeddings={emb.shape[0]} chunks={len(all_chunks)}"
        )

    print("Building FAISS index...")
    index = build_faiss_index(emb)

    # --- HARDENING: assert alignment after add
    if int(index.ntotal) != len(all_chunks):
        raise RuntimeError(
            f"BUG: faiss.ntotal != chunks after add: ntotal={int(index.ntotal)} chunks={len(all_chunks)}"
        )

    print(f"Saving FAISS index to {FAISS_PATH}")
    faiss.write_index(index, str(FAISS_PATH))

    # Also save a legacy-named copy so env INDEX_PATH=.../faiss.index stays correct.
    print(f"Saving legacy FAISS index to {FAISS_PATH_LEGACY}")
    faiss.write_index(index, str(FAISS_PATH_LEGACY))

    print(f"Saving metadata to {META_PATH}")
    with META_PATH.open("wb") as f:
        pickle.dump(all_chunks, f)

    # Config is read by app.py; keep old keys for backwards-compat
    config = {
        # Preferred key for app.py
        "embedding_model": EMBED_MODEL_NAME,
        # Backwards-compatible alias
        "embed_model": EMBED_MODEL_NAME,

        # Where the FAISS index actually lives
        "faiss_path": str(FAISS_PATH),

        # Default retrieval depth (kept in sync with app.py TOP_K)
        "top_k": 6,

        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_extractions": True,
        "chunk_extraction_version": 1,
        "files": [str(f.relative_to(ROOT)) for f in files],

        # Debug/helpful info
        "num_chunks": len(all_chunks),
        "faiss_ntotal": int(index.ntotal),
    }

    print(f"Saving config to {CONFIG_PATH}")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n✅ Ingest complete.")
    print(f"   Chunks written: {len(all_chunks)}")
    print(f"   FAISS ntotal:  {int(index.ntotal)}")
    print(f"   Index path:    {FAISS_PATH}")
    print(f"   Legacy path:   {FAISS_PATH_LEGACY}")
    print(f"   Meta path:     {META_PATH}")
    print(f"   Config path:   {CONFIG_PATH}")


if __name__ == "__main__":
    main()
