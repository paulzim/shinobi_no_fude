from ingest import simple_chunk_text
from scribe.text_seam import (
    build_extraction_context,
    build_grounded_prompt,
    extract_chunk_extractions,
)


def test_extract_chunk_extractions_for_rank_chunk():
    text = "\n".join(
        [
            "8th Kyu",
            "Weapon: Hanbo",
            "Kihon Happo: Ichimonji no Kata",
            "San Shin no Kata: Chi no Kata; Sui no Kata",
        ]
    )

    extractions = extract_chunk_extractions(text, "data/nttv rank requirements.txt")

    assert extractions["source_kind"] == "rank"
    assert "8th Kyu" in extractions["titles"]
    assert "Weapon: Hanbo" in extractions["anchors"]
    assert "Kihon Happo: Ichimonji no Kata" in extractions["anchors"]


def test_simple_chunk_text_persists_extractions_metadata():
    text = "\n".join(
        [
            "[WEAPON] Hanbo",
            "TYPE: Short staff",
            "RANKS: Introduced at 8th Kyu",
        ]
    )

    chunks = simple_chunk_text(text, source="data/NTTV Weapons Reference.txt")

    assert len(chunks) == 1
    meta = chunks[0]["meta"]
    assert meta["source_kind"] == "weapons"
    assert meta["extractions"]["source_kind"] == "weapons"
    assert "Weapon: Hanbo" in meta["extractions"]["anchors"]


def test_build_extraction_context_derives_from_unenriched_passages():
    passages = [
        {
            "text": "[WEAPON] Hanbo\nTYPE: Short staff\nRANKS: Introduced at 8th Kyu",
            "source": "data/NTTV Weapons Reference.txt",
            "meta": {"source": "data/NTTV Weapons Reference.txt"},
        }
    ]

    extraction_context = build_extraction_context(passages)

    assert "[Weapon] Weapon: Hanbo" in extraction_context
    assert "[Weapon] RANKS: Introduced at 8th Kyu" in extraction_context


def test_build_grounded_prompt_includes_extraction_context():
    prompt = build_grounded_prompt(
        context="[1] nttv rank requirements.txt\nWeapon: Hanbo",
        question="What weapon starts at 8th kyu?",
        extraction_context="- [Rank] 8th Kyu\n- [Rank] Weapon: Hanbo",
    )

    assert "Deterministic anchors derived from the retrieved sources:" in prompt
    assert "- [Rank] Weapon: Hanbo" in prompt
    assert "Question: What weapon starts at 8th kyu?" in prompt
