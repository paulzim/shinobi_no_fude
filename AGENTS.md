# AGENTS.md — shinobi_no_fude

This repo is a personal “writing studio” app (text + images) built on top of:
- deterministic extractors (accuracy anchors)
- RAG retrieval over a local corpus (FAISS + sidecar metadata)
- an external LLM for creative drafting (calls go out; the app runs locally)

## Non-negotiables
- Keep extractors deterministic and test-driven. If an extractor change breaks a test, fix the logic or update the test only with a clear reason.
- Prefer small, reviewable diffs. Avoid large refactors unless explicitly requested.
- Never invent or silently “correct” domain facts in extractors or the corpus. If a source is ambiguous, surface ambiguity.
- Preserve existing file/module names unless there’s a strong reason to change them (tests expect imports).

## Project layout (expected)
- `extractors/` — deterministic extractors + router entrypoint
- `tests/` — pytest suite; includes `tests/prompts/`
- `data/` — corpus sources
- `index/` — FAISS index + `meta.pkl` + `config.json`
- `scribe/` — new app code (pipeline/UI/writer/image seam), kept separate from legacy code

## Common commands
### Setup
- Create venv: `python -m venv .venv`
- Activate (Windows PowerShell): `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`

### Run tests
- `pytest -q`

### Run app (if Streamlit)
- `streamlit run app.py`

### Rebuild index (when asked)
- `python ingest.py`
(Only do this when explicitly requested; it can be slow and changes artifacts.)

## How to work in this repo (agent rules)
1. Before changing anything: identify the file(s) involved and skim relevant tests.
2. When modifying extractors/router: run pytest for the impacted tests (then full suite if reasonable).
3. When changing RAG/index loading: add a small smoke-test or at least a minimal validation step.
4. Avoid “drive-by formatting” changes. Only format touched areas.
5. If you add new modules under `scribe/`, keep interfaces explicit:
   - Text seam: extractors → RAG → writer prompt
   - Image seam: draft sections → prompt composer → backend adapter

## Quality bar
- No new warnings in logs for normal paths.
- Clear error messages for missing index/data/config.
- Defaults stay conservative (context caps, retrieval caps) unless user requests otherwise.