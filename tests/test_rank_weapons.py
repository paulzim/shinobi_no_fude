import pathlib

from extractors import try_extract_answer
from extractors.rank import try_answer_rank_weapons

RANK = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages_rank():
    return [{
        "text": RANK.read_text(encoding="utf-8"),
        "source": "nttv rank requirements.txt",
        "meta": {"priority": 3},
    }]


def test_rank_weapons_for_8th_kyu():
    q = "What weapon do I learn at 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "8th kyu" in low
    assert "weapon:" in low
    assert "hanbo" in low


def test_rank_weapons_for_5th_kyu():
    q = "What weapons are required for 5th kyu?"
    ans = try_extract_answer(q, _passages_rank())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "5th kyu" in low
    assert "weapon:" in low
    assert "knife" in low
    assert "shoto" in low


def test_rank_introducing_katana():
    q = "Which rank introduces katana?"
    ans = try_extract_answer(q, _passages_rank())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "katana" in low
    assert "7th kyu" in low


def test_rank_weapons_returns_none_when_rank_has_no_weapon():
    q = "What weapon do I learn at 9th kyu?"
    ans = try_answer_rank_weapons(q, _passages_rank())

    assert ans is None


def test_rank_weapons_returns_none_for_invalid_query():
    q = "What weapon should I practice next?"
    ans = try_answer_rank_weapons(q, _passages_rank())

    assert ans is None
