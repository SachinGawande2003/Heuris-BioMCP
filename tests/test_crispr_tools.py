from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_design_crispr_guides_scores_and_ranks_correctly(
    mock_http_client,
    mock_http_response,
):
    lookup_resp = mock_http_response(
        json_data={
            "Transcript": [
                {
                    "id": "ENST00000269305",
                    "is_canonical": 1,
                    "Exon": [{"id": "ENSE0001"}, {"id": "ENSE0002"}],
                }
            ]
        }
    )
    cds_resp = mock_http_response(text="ATGGCC" * 25)
    mock_http_client.get = AsyncMock(side_effect=[lookup_resp, cds_resp])

    candidate_sites = [
        {"guide": "AACCGGTTAACCGGTTAACC", "pam": "TGG", "strand": "+", "position": 6},
        {"guide": "GGGGGGGGGGGGGGGGGGGG", "pam": "AGG", "strand": "+", "position": 18},
        {"guide": "TTTTCCCCAAAAGGGGTTTT", "pam": "CGG", "strand": "-", "position": 30},
    ]

    def _fake_score_guide(seq: str, cas_variant: str, fractional_pos: float = 0.0):
        score_map = {
            "AACCGGTTAACCGGTTAACC": {
                "score": 88.0,
                "grade": "A",
                "gc_content_pct": 55.0,
                "seed_gc_pct": 58.0,
                "max_poly_T_run": 1,
                "u6_compatible": False,
                "restriction_sites": [],
                "feature_breakdown": {"gc_content": 15, "u6_compat": 0},
            },
            "GGGGGGGGGGGGGGGGGGGG": {
                "score": 72.0,
                "grade": "A-",
                "gc_content_pct": 100.0,
                "seed_gc_pct": 100.0,
                "max_poly_T_run": 0,
                "u6_compatible": True,
                "restriction_sites": ["BsmBI"],
                "feature_breakdown": {"gc_content": 4, "u6_compat": 10},
            },
            "TTTTCCCCAAAAGGGGTTTT": {
                "score": 61.0,
                "grade": "B",
                "gc_content_pct": 40.0,
                "seed_gc_pct": 42.0,
                "max_poly_T_run": 4,
                "u6_compatible": False,
                "restriction_sites": [],
                "feature_breakdown": {"gc_content": 10, "u6_compat": 0},
            },
        }
        return score_map[seq]

    with (
        patch("biomcp.tools.crispr_tools.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.crispr_tools._find_pam_sites", return_value=candidate_sites),
        patch("biomcp.tools.crispr_tools._score_guide", side_effect=_fake_score_guide),
    ):
        from biomcp.tools.crispr_tools import design_crispr_guides

        result = await design_crispr_guides.__wrapped__.__wrapped__.__wrapped__(
            "TP53",
            n_guides=2,
            min_score=50.0,
        )

    assert result["gene"] == "TP53"
    assert result["candidates_found"] == 3
    assert result["passing_min_score"] == 2
    assert len(result["guides"]) == 2
    assert [guide["rank"] for guide in result["guides"]] == [1, 2]
    assert [guide["score"] for guide in result["guides"]] == [88.0, 61.0]
    assert result["guides"][0]["sequence"] == "AACCGGTTAACCGGTTAACC"
    assert result["guides"][0]["ordering_sequence"] == "GAACCGGTTAACCGGTTAACC"
    assert all("BsmBI" not in guide["restriction_sites"] for guide in result["guides"])
