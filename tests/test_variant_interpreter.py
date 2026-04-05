from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_classify_variant_applies_pvs1_for_lof():
    vep_result = {
        "most_severe_consequence": "stop_gained",
        "predictions": {
            "SIFT": "deleterious",
            "PolyPhen2": "probably_damaging",
        },
        "splice_impact": False,
    }
    gnomad_result = {
        "allele_frequency": {
            "global_af": 0.0,
        }
    }

    with (
        patch("biomcp.tools.variant_interpreter._run_vep", new=AsyncMock(return_value=vep_result)),
        patch(
            "biomcp.tools.variant_interpreter._query_gnomad",
            new=AsyncMock(return_value=gnomad_result),
        ),
        patch(
            "biomcp.tools.variant_interpreter._query_clinvar",
            new=AsyncMock(return_value={}),
        ),
    ):
        from biomcp.tools.variant_interpreter import classify_variant

        result = await classify_variant.__wrapped__.__wrapped__.__wrapped__(
            "TP53",
            "c.524G>A",
            consequence="stop_gained",
        )

    evidence_codes = {item["code"] for item in result["evidence_codes"]}

    assert result["classification"] == "Pathogenic"
    assert result["acmg_class"] == 5
    assert result["acmg_score"] == 11
    assert {"PVS1", "PM2", "PP3"} <= evidence_codes
    assert "Loss-of-function: stop_gained" in result["key_findings"]
