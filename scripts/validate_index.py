from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = REPO_ROOT / "index"
TOP_K = 5
SEED_CHECKS = [
    (
        "rank",
        "8th kyu + hanbo",
        ("8th kyu", "hanbo"),
        ("rank requirements",),
        ("rank requirements",),
    ),
    (
        "weapons",
        "katana",
        ("katana",),
        ("weapons", "katana"),
        ("weapons",),
    ),
    (
        "glossary",
        "buyu",
        ("buyu",),
        ("buyu", "glossary"),
        ("buyu", "glossary"),
    ),
]


def _load_config(index_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(os.getenv("CONFIG_PATH", str(index_dir / "config.json")))
    meta_path = Path(os.getenv("META_PATH", str(index_dir / "meta.pkl")))

    if not config_path.exists():
        raise RuntimeError(f"Missing config.json: {config_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing meta.pkl: {meta_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg, config_path, meta_path


def _faiss_candidates(index_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []

    idx_env = os.getenv("INDEX_PATH")
    if idx_env:
        candidates.append(Path(idx_env))

    cfg_faiss = cfg.get("faiss_path")
    if cfg_faiss:
        cfg_path = Path(cfg_faiss)
        candidates.append(cfg_path if cfg_path.is_absolute() else index_dir / cfg_path)

    candidates.append(index_dir / "index.faiss")
    candidates.append(index_dir / "faiss.index")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _load_chunks(meta_path: Path) -> list[dict[str, Any]]:
    with meta_path.open("rb") as fh:
        chunks = pickle.load(fh)
    if not isinstance(chunks, list):
        raise RuntimeError(f"meta.pkl did not contain a list: {type(chunks).__name__}")
    return chunks


def _choose_index(
    candidates: list[Path],
    meta_path: Path,
) -> tuple[Path, Any, list[dict[str, Any]], list[Path]]:
    tried: list[Path] = []

    for candidate in candidates:
        tried.append(candidate)
        if not candidate.exists():
            continue

        index = faiss.read_index(str(candidate))
        chunks = _load_chunks(meta_path)
        ntotal = int(getattr(index, "ntotal", 0) or 0)
        if ntotal <= 0 or len(chunks) <= 0:
            continue
        if ntotal != len(chunks):
            continue
        return candidate, index, chunks, tried

    tried_text = "\n".join(f"- {path}" for path in tried)
    raise RuntimeError(
        "No usable FAISS index matched meta.pkl.\n"
        f"Candidates tried:\n{tried_text}"
    )


def _source_for(chunk: dict[str, Any]) -> str:
    source = chunk.get("source")
    if isinstance(source, str) and source:
        return source
    meta = chunk.get("meta") or {}
    if isinstance(meta, dict):
        inner_source = meta.get("source")
        if isinstance(inner_source, str) and inner_source:
            return inner_source
    return "<unknown>"


def _preview(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _find_seed_index(
    chunks: list[dict[str, Any]],
    terms: tuple[str, ...],
    preferred_sources: tuple[str, ...],
) -> int:
    lowered = tuple(term.lower() for term in terms)
    preferred = tuple(term.lower() for term in preferred_sources)

    for idx, chunk in enumerate(chunks):
        source = _source_for(chunk).lower()
        haystack = " ".join([source, str(chunk.get("text", ""))]).lower()
        if all(term in haystack for term in lowered) and any(term in source for term in preferred):
            return idx

    for idx, chunk in enumerate(chunks):
        haystack = " ".join(
            [
                _source_for(chunk),
                str(chunk.get("text", "")),
            ]
        ).lower()
        if all(term in haystack for term in lowered):
            return idx
    raise RuntimeError(f"Could not find seed chunk for terms: {terms}")


def _print_retrieval_checks(
    index: Any,
    chunks: list[dict[str, Any]],
) -> list[str]:
    want = min(TOP_K, int(getattr(index, "ntotal", 0) or 0))

    failures: list[str] = []
    print("\nRetrieval sanity checks (offline seed-vector mode)")
    for label, description, seed_terms, expected_terms, preferred_sources in SEED_CHECKS:
        seed_idx = _find_seed_index(chunks, seed_terms, preferred_sources)
        seed_chunk = chunks[seed_idx]
        seed_vector = np.asarray(index.reconstruct(seed_idx), dtype="float32")[None, :]
        distances, indices = index.search(seed_vector, want)

        print(f"\n[{label}] seed terms: {description}")
        print(f"  seed_source={_source_for(seed_chunk)}")
        print(f"  seed_preview={_preview(str(seed_chunk.get('text', '')))}")
        seen_sources: set[str] = set()
        unique_sources: list[str] = []

        for rank, (chunk_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                continue
            chunk = chunks[chunk_idx]
            source = _source_for(chunk)
            preview = _preview(str(chunk.get("text", "")))
            print(f"  {rank}. score={float(score):.4f} source={source}")
            print(f"     preview={preview}")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)

        source_text = " | ".join(unique_sources).lower()
        if not any(term in source_text for term in expected_terms):
            failures.append(
                f"{label}: none of the top sources matched expected terms {expected_terms}"
            )

    return failures


def main() -> int:
    index_dir = Path(os.getenv("INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    cfg, config_path, meta_path = _load_config(index_dir)
    candidates = _faiss_candidates(index_dir, cfg)
    chosen_path, index, chunks, tried = _choose_index(candidates, meta_path)

    ntotal = int(getattr(index, "ntotal", 0) or 0)
    model_name = (
        cfg.get("embedding_model")
        or cfg.get("embed_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Index artifact validation")
    print(f"repo_root:   {REPO_ROOT}")
    print(f"index_dir:   {index_dir}")
    print(f"config_path: {config_path}")
    print(f"meta_path:   {meta_path}")
    print(f"faiss_path:  {chosen_path}")
    print(f"embed_model: {model_name}")
    if chosen_path.parent != index_dir:
        print("note: selected FAISS path is outside index_dir because config.json points there")

    print("\nFAISS candidates")
    for candidate in candidates:
        marker = "SELECTED" if candidate == chosen_path else "checked"
        exists = "exists" if candidate.exists() else "missing"
        print(f"- {candidate} [{exists}; {marker}]")

    print("\nConsistency checks")
    print(f"- faiss.ntotal={ntotal}")
    print(f"- len(meta.pkl)={len(chunks)}")

    cfg_num_chunks = cfg.get("num_chunks")
    if cfg_num_chunks is not None:
        print(f"- config.num_chunks={cfg_num_chunks}")

    cfg_ntotal = cfg.get("faiss_ntotal")
    if cfg_ntotal is not None:
        print(f"- config.faiss_ntotal={cfg_ntotal}")

    failures: list[str] = []
    if ntotal != len(chunks):
        failures.append(f"faiss.ntotal ({ntotal}) != len(meta.pkl) ({len(chunks)})")
    if cfg_num_chunks is not None and int(cfg_num_chunks) != len(chunks):
        failures.append(
            f"config.num_chunks ({cfg_num_chunks}) != len(meta.pkl) ({len(chunks)})"
        )
    if cfg_ntotal is not None and int(cfg_ntotal) != ntotal:
        failures.append(f"config.faiss_ntotal ({cfg_ntotal}) != faiss.ntotal ({ntotal})")

    failures.extend(_print_retrieval_checks(index, chunks))

    if failures:
        print("\nValidation failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
