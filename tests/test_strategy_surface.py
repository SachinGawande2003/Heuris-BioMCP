from __future__ import annotations

import pytest

from biomcp.tools import strategy_surface
from biomcp.utils import get_cache


@pytest.fixture(autouse=True)
def clear_strategy_caches() -> None:
    for namespace in ("default", "uniprot", "reactome", "kegg", "fda", "pharmgkb", "omim", "ensembl"):
        get_cache(namespace).clear()


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeGenomeClient:
    async def get(self, url: str, params=None, headers=None):
        if "/lookup/symbol/homo_sapiens/TP53" in url:
            return FakeResponse({"seq_region_name": "17", "start": 7668402, "end": 7687550})
        if "/overlap/region/human/" in url:
            feature = (params or {}).get("feature")
            if feature == "gene":
                return FakeResponse(
                    [
                        {
                            "id": "ENSG00000141510",
                            "external_name": "TP53",
                            "biotype": "protein_coding",
                            "start": 7668402,
                            "end": 7687550,
                            "strand": 1,
                        }
                    ]
                )
            if feature == "variation":
                return FakeResponse(
                    [
                        {
                            "id": "rs28934578",
                            "start": 7673803,
                            "end": 7673803,
                            "strand": 1,
                            "consequence_type": ["missense_variant"],
                        }
                    ]
                )
            if feature == "regulatory":
                return FakeResponse(
                    [
                        {
                            "id": "ENSR00000000001",
                            "feature_type": "promoter",
                            "description": "Promoter-like signature",
                            "start": 7676500,
                            "end": 7676900,
                        }
                    ]
                )
        raise AssertionError(f"Unexpected URL: {url}")


class FakePubChemClient:
    async def get(self, url: str, params=None, headers=None):
        if "/compound/name/" in url:
            return FakeResponse({}, status_code=404)
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.mark.asyncio
async def test_find_protein_combines_uniprot_and_pdb(monkeypatch: pytest.MonkeyPatch):
    import biomcp.tools.proteins as proteins

    async def fake_search_proteins(query: str, organism: str, max_results: int, reviewed_only: bool):
        return {"proteins": [{"accession": "P04637", "name": "Cellular tumor antigen p53"}]}

    async def fake_search_pdb(query: str, max_results: int):
        return {"structures": [{"pdb_id": "1TUP"}]}

    monkeypatch.setattr(proteins, "search_proteins", fake_search_proteins)
    monkeypatch.setattr(proteins, "search_pdb_structures", fake_search_pdb)

    result = await strategy_surface.find_protein(query="TP53", source="both", max_results=3)
    assert result["uniprot_results"]["proteins"][0]["accession"] == "P04637"
    assert result["pdb_results"]["structures"][0]["pdb_id"] == "1TUP"


@pytest.mark.asyncio
async def test_pathway_analysis_gene_context_uses_kegg_gene_membership(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.pathways as pathways

    async def fake_reactome(gene_symbol: str):
        return {"gene": gene_symbol, "total": 2, "pathways": [{"reactome_id": "R-HSA-177929"}]}

    async def fake_kegg_gene_pathways(gene_symbol: str, organism: str = "hsa"):
        return {
            "gene": gene_symbol,
            "organism": organism,
            "ncbi_gene_id": "1956",
            "kegg_gene_ids": ["hsa:1956"],
            "total": 3,
            "pathways": [{"pathway_id": "hsa04010"}],
        }

    monkeypatch.setattr(pathways, "get_reactome_pathways", fake_reactome)
    monkeypatch.setattr(pathways, "get_kegg_gene_pathways", fake_kegg_gene_pathways)

    result = await strategy_surface.pathway_analysis(action="gene_context", gene_symbol="EGFR")

    assert result["reactome"]["total"] == 2
    assert result["kegg"]["ncbi_gene_id"] == "1956"
    assert result["kegg"]["pathways"][0]["pathway_id"] == "hsa04010"


@pytest.mark.asyncio
async def test_boltz2_workflow_protein_ligand_respects_predict_affinity(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.nvidia_nim as nvidia_nim

    captured: dict[str, object] = {}

    async def fake_design_protein_ligand(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(nvidia_nim, "design_protein_ligand", fake_design_protein_ligand)

    result = await strategy_surface.boltz2_workflow(
        mode="protein_ligand",
        uniprot_accession="P04637",
        ligand_smiles=["CCO"],
        predict_affinity=False,
    )

    assert result["status"] == "ok"
    assert captured["predict_affinity"] is False


@pytest.mark.asyncio
async def test_biomarker_panel_design_merges_open_targets_and_heuristic(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_open_targets(disease: str, panel_size: int):
        return [
            {
                "gene": "EGFR",
                "gene_name": "epidermal growth factor receptor",
                "score": 0.95,
                "evidence_breakdown": [{"id": "literature", "score": 0.8}],
                "source": "Open Targets",
            }
        ]

    async def fake_heuristic(disease: str, panel_size: int):
        return [
            {
                "gene": "EGFR",
                "gene_name": "epidermal growth factor receptor",
                "score": 0.60,
                "evidence_breakdown": [{"id": "pubmed_frequency", "score": 5}],
                "source": "PubMed heuristic",
            },
            {
                "gene": "ERBB2",
                "gene_name": "erb-b2 receptor tyrosine kinase 2",
                "score": 0.70,
                "evidence_breakdown": [{"id": "pubmed_frequency", "score": 4}],
                "source": "PubMed heuristic",
            },
            {
                "gene": "MET",
                "gene_name": "MET proto-oncogene",
                "score": 0.65,
                "evidence_breakdown": [{"id": "pubmed_frequency", "score": 3}],
                "source": "PubMed heuristic",
            },
        ]

    monkeypatch.setattr(strategy_surface, "_search_open_targets_disease", fake_open_targets)
    monkeypatch.setattr(strategy_surface, "_heuristic_biomarker_candidates", fake_heuristic)

    result = await strategy_surface.biomarker_panel_design("lung cancer", panel_size=3)
    genes = [entry["gene"] for entry in result["panel"]]

    assert genes == ["EGFR", "ERBB2", "MET"]
    assert result["evidence_source"] == "Open Targets + PubMed heuristic"
    assert result["panel_coverage"]["returned"] == 3


@pytest.mark.asyncio
async def test_pharmacogenomics_report_uses_cpic_and_label_mentions(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.databases as databases
    import biomcp.tools.drug_safety as drug_safety

    async def fake_label(drug_name: str):
        return {
            "warnings_and_cautions": "Consider DPYD deficiency and UGT1A1 testing before therapy.",
            "use_in_specific_populations": "",
            "boxed_warning": "",
            "drug_interactions": "",
            "pharmacogenomics": "UGT1A1 variants affect toxicity.",
        }

    async def fake_pharmgkb(gene_symbol: str, max_results: int = 10):
        return {
            "pharmgkb_gene_id": f"PGX-{gene_symbol}",
            "clinical_annotations": [{"drugs": ["irinotecan"], "evidence_level": "1A"}],
        }

    monkeypatch.setattr(drug_safety, "get_drug_label_warnings", fake_label)
    monkeypatch.setattr(databases, "get_pharmgkb_variants", fake_pharmgkb)

    result = await strategy_surface.pharmacogenomics_report("irinotecan")

    assert "UGT1A1" in result["genes_considered"]
    assert "DPYD" in result["genes_considered"]
    assert "DPYD" in result["label_gene_mentions"]
    assert result["testing_recommendations"]


@pytest.mark.asyncio
async def test_protein_family_analysis_includes_interpro_domain_architecture(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.innovations as innovations
    import biomcp.tools.proteins as proteins

    async def fake_get_protein_info(accession: str):
        return {
            "full_name": "Epidermal growth factor receptor",
            "gene_names": ["EGFR"],
            "features": [{"type": "domain", "description": "Kinase domain", "start": 712, "end": 979}],
        }

    async def fake_get_protein_domain_structure(uniprot_accession: str, include_disordered: bool = False):
        return {
            "domain_coverage_pct": 62.5,
            "domain_diagram": "1----████----1210",
            "domains": [{"name": "Protein kinase domain", "start": 712, "end": 979}],
        }

    monkeypatch.setattr(proteins, "get_protein_info", fake_get_protein_info)
    monkeypatch.setattr(innovations, "get_protein_domain_structure", fake_get_protein_domain_structure)

    result = await strategy_surface.protein_family_analysis(accession="P00533")

    assert result["domain_coverage_pct"] == 62.5
    assert result["domain_architecture"][0]["name"] == "Protein kinase domain"
    assert "Protein" in result["putative_family_keywords"]


@pytest.mark.asyncio
async def test_network_enrichment_adds_input_gene_edges_and_pathway_enrichment(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.databases as databases
    import biomcp.tools.innovations as innovations
    import biomcp.tools.pathways as pathways

    async def fake_reactome(gene_symbol: str):
        return {"pathways": [{"name": "MAPK signaling", "reactome_id": "R-HSA-123"}]}

    async def fake_string(gene_symbol: str, min_score: int = 700, max_results: int = 10):
        interactions = {
            "KRAS": [{"partner": "BRAF", "combined_score": 0.97}, {"partner": "RAF1", "combined_score": 0.91}],
            "BRAF": [{"partner": "KRAS", "combined_score": 0.97}, {"partner": "MAP2K1", "combined_score": 0.92}],
        }
        return {"gene": gene_symbol, "interactions": interactions[gene_symbol]}

    async def fake_enrichment(gene_list: list[str], database: str = "both", min_genes: int = 2, **kwargs):
        return {
            "pathways_tested": 4,
            "significant_pathways": 1,
            "enriched_pathways": [{"pathway_name": "MAPK signaling", "fdr": 0.01}],
        }

    monkeypatch.setattr(pathways, "get_reactome_pathways", fake_reactome)
    monkeypatch.setattr(databases, "get_string_interactions", fake_string)
    monkeypatch.setattr(innovations, "compute_pathway_enrichment", fake_enrichment)

    result = await strategy_surface.network_enrichment(["KRAS", "BRAF"], max_results=5)

    assert result["input_gene_edges"][0]["gene_a"] == "BRAF"
    assert result["input_gene_edges"][0]["gene_b"] == "KRAS"
    assert result["pathway_enrichment"][0]["pathway_name"] == "MAPK signaling"


@pytest.mark.asyncio
async def test_drug_interaction_checker_detects_bidirectional_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.drug_safety as drug_safety

    async def fake_label(drug_name: str):
        if drug_name.lower() == "warfarin":
            return {
                "drug_interactions": "Avoid coadministration with fluconazole.",
                "boxed_warning": "",
                "warnings_and_cautions": "",
                "contraindications": "",
                "full_label_url": "https://example.com/warfarin",
                "has_black_box_warning": False,
            }
        return {
            "drug_interactions": "May increase exposure to warfarin.",
            "boxed_warning": "Serious bleeding risk with warfarin.",
            "warnings_and_cautions": "",
            "contraindications": "",
            "full_label_url": "https://example.com/fluconazole",
            "has_black_box_warning": True,
        }

    monkeypatch.setattr(drug_safety, "get_drug_label_warnings", fake_label)

    result = await strategy_surface.drug_interaction_checker("warfarin", ["fluconazole"])
    interaction = result["interactions"][0]

    assert interaction["detected_in_primary_label"] is True
    assert interaction["detected_in_secondary_label"] is True
    assert interaction["severity"] == "high"


@pytest.mark.asyncio
async def test_structural_similarity_returns_note_when_query_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_get_http_client():
        return FakePubChemClient()

    monkeypatch.setattr(strategy_surface, "get_http_client", fake_get_http_client)

    result = await strategy_surface.structural_similarity(query="unknown compound")
    assert result["matches"] == []
    assert "No PubChem compound could be resolved" in result["note"]


@pytest.mark.asyncio
async def test_rare_disease_diagnosis_combines_omim_and_open_targets(
    monkeypatch: pytest.MonkeyPatch,
):
    import biomcp.tools.databases as databases
    import biomcp.tools.pathways as pathways

    async def fake_omim(gene_symbol: str):
        return {
            "diseases": [
                {
                    "omim_id": "151623",
                    "phenotype": "Li-Fraumeni syndrome",
                    "inheritance_pattern": "Autosomal Dominant (AD)",
                    "omim_url": "https://omim.org/entry/151623",
                }
            ]
        }

    async def fake_open_targets(gene_symbol: str, max_results: int = 10):
        return {
            "associations": [
                {
                    "disease_name": "Hereditary cancer-predisposing syndrome",
                    "description": "Cancer predisposition syndrome with sarcoma and breast cancer risk.",
                    "overall_score": 0.82,
                    "therapeutic_areas": ["rare genetic disease"],
                    "url": "https://platform.opentargets.org/example",
                }
            ]
        }

    monkeypatch.setattr(databases, "get_omim_gene_diseases", fake_omim)
    monkeypatch.setattr(pathways, "get_gene_disease_associations", fake_open_targets)

    result = await strategy_surface.rare_disease_diagnosis(
        phenotype_terms=["sarcoma", "breast cancer"],
        gene_symbol="TP53",
    )

    assert result["differential_diagnosis"][0]["source"] == "Open Targets"
    assert any(entry["source"] == "OMIM" for entry in result["differential_diagnosis"])


@pytest.mark.asyncio
async def test_genome_browser_snapshot_includes_region_feature_summary(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_get_http_client():
        return FakeGenomeClient()

    monkeypatch.setattr(strategy_surface, "get_http_client", fake_get_http_client)

    result = await strategy_surface.genome_browser_snapshot(gene_symbol="TP53", flank_bp=5000)

    assert result["region"].startswith("17:")
    assert result["region_feature_summary"]["counts"]["genes"] == 1
    assert result["region_feature_summary"]["notable_variants"][0]["variant_id"] == "rs28934578"
