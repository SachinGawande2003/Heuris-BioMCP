from __future__ import annotations

import pytest

import biomcp.tools.innovations as innovations
import biomcp.utils as utils
from biomcp.tools.innovations import bulk_gene_analysis


@pytest.mark.asyncio
async def test_bulk_gene_analysis_differential_mode(monkeypatch: pytest.MonkeyPatch):
    import biomcp.tools.ncbi as ncbi_module
    import biomcp.tools.pathways as pathway_module

    disease_map = {
        "EGFR": ["Lung cancer", "Glioblastoma"],
        "ERBB2": ["Lung cancer", "Breast cancer"],
        "MET": ["Lung cancer"],
        "BRCA1": ["Breast cancer", "DNA repair deficiency"],
        "BRCA2": ["Breast cancer", "DNA repair deficiency"],
        "PALB2": ["DNA repair deficiency"],
    }
    pathway_map = {
        "EGFR": ["ERBB signaling", "MAPK cascade"],
        "ERBB2": ["ERBB signaling", "PI3K signaling"],
        "MET": ["ERBB signaling"],
        "BRCA1": ["Homologous recombination", "DNA damage checkpoint"],
        "BRCA2": ["Homologous recombination", "DNA damage checkpoint"],
        "PALB2": ["Homologous recombination"],
    }

    async def fake_get_gene_info(gene_symbol: str) -> dict[str, str]:
        return {"symbol": gene_symbol}

    async def fake_get_drug_targets(gene_symbol: str, max_results: int = 10) -> dict[str, object]:
        return {"drugs": [{"molecule_name": f"{gene_symbol}-drug"}]}

    async def fake_get_gene_disease_associations(
        gene_symbol: str,
        max_results: int = 12,
    ) -> dict[str, object]:
        names = disease_map[gene_symbol][:max_results]
        return {
            "total_associations": len(names),
            "associations": [{"disease_name": name} for name in names],
        }

    async def fake_get_reactome_pathways(gene_symbol: str) -> dict[str, object]:
        names = pathway_map[gene_symbol]
        return {
            "total": len(names),
            "pathways": [{"name": name} for name in names],
        }

    monkeypatch.setattr(ncbi_module, "get_gene_info", fake_get_gene_info)
    monkeypatch.setattr(pathway_module, "get_drug_targets", fake_get_drug_targets)
    monkeypatch.setattr(
        pathway_module,
        "get_gene_disease_associations",
        fake_get_gene_disease_associations,
    )
    monkeypatch.setattr(pathway_module, "get_reactome_pathways", fake_get_reactome_pathways)

    result = await bulk_gene_analysis(
        gene_symbols=["EGFR", "ERBB2", "MET"],
        reference_gene_symbols=["BRCA1", "BRCA2", "PALB2"],
        group_a_label="tumor_panel",
        group_b_label="repair_panel",
        comparison_axes=["diseases", "pathways"],
    )

    assert result["mode"] == "differential_panel_analysis"
    assert result["primary_genes"] == ["EGFR", "ERBB2", "MET"]
    assert result["reference_genes"] == ["BRCA1", "BRCA2", "PALB2"]

    top_pathway = result["differential_analysis"]["pathways"]["group_a_enriched"][0]
    assert top_pathway["pathway"] == "ERBB signaling"
    assert top_pathway["enriched_in"] == "tumor_panel"
    assert top_pathway["group_a_hits"] == 3
    assert top_pathway["group_b_hits"] == 0

    top_disease = result["differential_analysis"]["diseases"]["group_b_enriched"][0]
    assert top_disease["disease"] == "DNA repair deficiency"
    assert top_disease["enriched_in"] == "repair_panel"
    assert top_disease["group_b_hits"] == 3


@pytest.mark.asyncio
async def test_bulk_gene_analysis_preserves_legacy_mode(monkeypatch: pytest.MonkeyPatch):
    import biomcp.tools.ncbi as ncbi_module
    import biomcp.tools.pathways as pathway_module

    async def fake_get_gene_info(gene_symbol: str) -> dict[str, str]:
        return {"symbol": gene_symbol}

    async def fake_get_drug_targets(gene_symbol: str, max_results: int = 10) -> dict[str, object]:
        return {"drugs": [{"molecule_name": "DrugA"}, {"molecule_name": "DrugB"}]}

    async def fake_get_gene_disease_associations(
        gene_symbol: str,
        max_results: int = 12,
    ) -> dict[str, object]:
        return {
            "total_associations": 1,
            "associations": [{"disease_name": "Shared disease"}],
        }

    async def fake_get_reactome_pathways(gene_symbol: str) -> dict[str, object]:
        return {
            "total": 1,
            "pathways": [{"name": "Shared pathway"}],
        }

    monkeypatch.setattr(ncbi_module, "get_gene_info", fake_get_gene_info)
    monkeypatch.setattr(pathway_module, "get_drug_targets", fake_get_drug_targets)
    monkeypatch.setattr(
        pathway_module,
        "get_gene_disease_associations",
        fake_get_gene_disease_associations,
    )
    monkeypatch.setattr(pathway_module, "get_reactome_pathways", fake_get_reactome_pathways)

    result = await bulk_gene_analysis(gene_symbols=["EGFR", "ERBB2"])

    assert result["mode"] == "comparative_summary"
    assert result["differential_analysis"] is None
    assert result["comparison_matrix"]["shared_diseases"]["Shared disease"] == ["EGFR", "ERBB2"]
    assert result["comparison_matrix"]["shared_pathways"]["Shared pathway"] == ["EGFR", "ERBB2"]


@pytest.mark.asyncio
async def test_get_protein_domain_structure_matches_lowercase_isoform_accessions(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200):
            self._payload = payload
            self.status_code = status_code

        def json(self) -> dict[str, object]:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class FakeClient:
        async def get(self, url: str, headers=None, params=None):
            if "/entry/interpro/protein/uniprot/" in url:
                return FakeResponse(
                    {
                        "results": [
                            {
                                "metadata": {
                                    "accession": "IPR000719",
                                    "name": "Protein kinase domain",
                                    "type": "domain",
                                    "source_database": "InterPro",
                                    "description": "Catalytic kinase domain",
                                    "go_terms": [{"identifier": "GO:0004672", "name": "protein kinase activity"}],
                                },
                                "proteins": [
                                    {
                                        "accession": "p00533-2",
                                        "entry_protein_locations": [
                                            {"fragments": [{"start": 712, "end": 979}]}
                                        ],
                                    },
                                    {
                                        "uniprot_accession": "P00533",
                                        "entry_protein_locations": [
                                            {"fragments": [{"start": 1000, "end": 1050}]}
                                        ],
                                    },
                                ],
                            }
                        ]
                    }
                )
            if "/protein/uniprot/" in url:
                return FakeResponse({"metadata": {"length": 1210}})
            raise AssertionError(f"Unexpected URL: {url}")

    async def fake_get_http_client() -> FakeClient:
        return FakeClient()

    monkeypatch.setattr(innovations, "get_http_client", fake_get_http_client)
    utils.get_cache("interpro").clear()

    result = await innovations.get_protein_domain_structure("P00533")

    assert result["total_domains"] == 1
    assert result["domains"][0]["name"] == "Protein kinase domain"
    assert result["domains"][0]["start"] == 712
    assert result["domain_coverage_pct"] == 22.1
