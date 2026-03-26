"""
BioMCP — Pathway & Drug Discovery Tools
========================================
Tools:
  search_pathways              — KEGG pathway search
  get_pathway_genes            — all genes in a KEGG pathway
  get_reactome_pathways        — Reactome pathways for a gene
  get_drug_targets             — ChEMBL drug-target activities
  get_compound_info            — ChEMBL compound details + ADMET
  get_gene_disease_associations— Open Targets gene-disease evidence

APIs:
  KEGG REST       https://rest.kegg.jp/
  Reactome        https://reactome.org/ContentService/
  ChEMBL REST     https://www.ebi.ac.uk/chembl/api/data/
  Open Targets    https://api.platform.opentargets.org/api/v4/graphql
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

KEGG_BASE         = "https://rest.kegg.jp"
REACTOME_BASE     = "https://reactome.org/ContentService"
CHEMBL_BASE       = "https://www.ebi.ac.uk/chembl/api/data"
OPENTARGETS_GQL   = "https://api.platform.opentargets.org/api/v4/graphql"


# ─────────────────────────────────────────────────────────────────────────────
# KEGG — pathway search
# ─────────────────────────────────────────────────────────────────────────────

@cached("kegg")
@rate_limited("kegg")
@with_retry(max_attempts=3)
async def search_pathways(query: str, organism: str = "hsa") -> dict[str, Any]:
    """
    Search KEGG for biological pathways.

    Args:
        query:    Keyword — pathway name, gene, disease (e.g. 'apoptosis', 'EGFR').
        organism: KEGG organism code. Default 'hsa' (human).
                  Other: 'mmu' mouse · 'rno' rat · 'dme' fly · 'sce' yeast.

    Returns:
        { query, organism, total, pathways: [{ pathway_id, organism_id,
          description, viewer_url, image_url }] }
    """
    client = await get_http_client()

    resp = await client.get(
        f"{KEGG_BASE}/find/pathway/{query}",
        headers={"Accept": "text/plain"},
    )
    if resp.status_code == 404:
        return {"query": query, "organism": organism, "total": 0, "pathways": []}
    resp.raise_for_status()

    pathways: list[dict[str, Any]] = []
    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        path_id, desc = parts
        code    = path_id.replace("path:", "").strip()
        org_id  = code.replace("map", organism)
        pathways.append({
            "pathway_id":   code,
            "organism_id":  org_id,
            "description":  desc.strip(),
            "viewer_url":   f"https://www.kegg.jp/pathway/{org_id}",
            "image_url":    f"https://www.kegg.jp/kegg/pathway/{org_id}/{org_id}.png",
        })

    return {"query": query, "organism": organism, "total": len(pathways), "pathways": pathways}


# ─────────────────────────────────────────────────────────────────────────────
# KEGG — genes in a pathway
# ─────────────────────────────────────────────────────────────────────────────

@cached("kegg")
@rate_limited("kegg")
@with_retry(max_attempts=3)
async def get_pathway_genes(pathway_id: str) -> dict[str, Any]:
    """
    List all genes in a KEGG pathway.

    Args:
        pathway_id: KEGG pathway ID (e.g. 'hsa05200', 'hsa04010').

    Returns:
        { pathway_id, total_genes, genes: [{ kegg_id, symbol, description }] }
    """
    pathway_id = BioValidator.validate_kegg_pathway_id(pathway_id)
    client     = await get_http_client()

    # Gene ID links
    link_resp = await client.get(f"{KEGG_BASE}/link/genes/{pathway_id}")
    if link_resp.status_code == 404:
        return {"pathway_id": pathway_id, "total_genes": 0, "genes": []}
    link_resp.raise_for_status()

    gene_ids = [
        line.split("\t")[1].strip()
        for line in link_resp.text.strip().split("\n")
        if "\t" in line
    ]

    genes: list[dict[str, str]] = []
    # Batch in groups of 10 (KEGG API limit)
    for i in range(0, min(len(gene_ids), 60), 10):
        batch = gene_ids[i:i + 10]
        info_resp = await client.get(f"{KEGG_BASE}/get/{'+'.join(batch)}")
        if info_resp.status_code != 200:
            continue
        for block in info_resp.text.split("///"):
            block = block.strip()
            if not block:
                continue
            rec: dict[str, str] = {}
            for line in block.split("\n"):
                if line.startswith("ENTRY"):
                    rec["kegg_id"] = line.split()[1]
                elif line.startswith("NAME"):
                    rec["symbol"] = line.replace("NAME", "").strip().split(",")[0]
                elif line.startswith("DEFINITION"):
                    rec["description"] = line.replace("DEFINITION", "").strip()
            if rec:
                genes.append(rec)

    return {"pathway_id": pathway_id, "total_genes": len(gene_ids), "genes": genes}


# ─────────────────────────────────────────────────────────────────────────────
# Reactome — pathway mapping for a gene
# ─────────────────────────────────────────────────────────────────────────────

@cached("reactome")
@rate_limited("reactome")
@with_retry(max_attempts=3)
async def get_reactome_pathways(
    gene_symbol: str,
    species: str = "9606",
) -> dict[str, Any]:
    """
    Get Reactome pathways associated with a gene.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'TP53').
        species:     NCBI taxonomy ID (default '9606' = Homo sapiens).

    Returns:
        { gene, species_taxid, total, pathways: [{ reactome_id, name,
          type, species, url, diagram_url }] }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    client      = await get_http_client()

    map_resp = await client.post(
        f"{REACTOME_BASE}/identifiers/mapping",
        content=gene_symbol,
        headers={"Content-Type": "text/plain", "Accept": "application/json"},
    )
    if map_resp.status_code == 404:
        return {"gene": gene_symbol, "total": 0, "pathways": [],
                "note": f"'{gene_symbol}' not found in Reactome."}
    map_resp.raise_for_status()

    pathways: list[dict[str, str]] = []
    for result in map_resp.json().get("results", []):
        for entry in result.get("entries", []):
            stid = entry.get("stId", "")
            if not stid:
                continue
            pathways.append({
                "reactome_id": stid,
                "name":        entry.get("name", ""),
                "type":        entry.get("type", ""),
                "species":     entry.get("species", {}).get("name", ""),
                "url":         f"https://reactome.org/content/detail/{stid}",
                "diagram_url": f"https://reactome.org/PathwayBrowser/#/{stid}",
            })

    return {
        "gene":         gene_symbol,
        "species_taxid":species,
        "total":        len(pathways),
        "pathways":     pathways,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL — drug-target activity profiles
# ─────────────────────────────────────────────────────────────────────────────

@cached("drug_target")
@rate_limited("chembl")
@with_retry(max_attempts=3)
async def get_drug_targets(
    gene_symbol: str,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Find compounds targeting a gene from ChEMBL.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'EGFR', 'BRAF', 'KRAS').
        max_results: Max compound entries (1–100). Default 20.

    Returns:
        { gene, target_chembl_id, target_name, total_activities,
          drugs: [{ molecule_chembl_id, molecule_name, activity_type,
                    activity_value, activity_units, evalue, chembl_url }] }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    # Step 1 — find target
    tgt_resp = await client.get(
        f"{CHEMBL_BASE}/target/search.json",
        params={"q": gene_symbol, "organism": "Homo sapiens", "limit": 5},
    )
    tgt_resp.raise_for_status()
    targets = tgt_resp.json().get("targets", [])

    if not targets:
        return {"gene": gene_symbol, "drugs": [],
                "error": f"Target '{gene_symbol}' not found in ChEMBL."}

    target_id   = targets[0].get("target_chembl_id", "")
    target_name = targets[0].get("pref_name", "")

    # Step 2 — get activities
    act_resp = await client.get(
        f"{CHEMBL_BASE}/activity.json",
        params={
            "target_chembl_id":   target_id,
            "standard_type__in":  "IC50,Ki,Kd,EC50,GI50",
            "limit":              max_results,
            "order_by":           "standard_value",
        },
    )
    act_resp.raise_for_status()
    act_data = act_resp.json()

    seen: set[str] = set()
    drugs: list[dict[str, Any]] = []
    for act in act_data.get("activities", []):
        mol_id = act.get("molecule_chembl_id", "")
        if not mol_id or mol_id in seen:
            continue
        seen.add(mol_id)
        drugs.append({
            "molecule_chembl_id": mol_id,
            "molecule_name":      act.get("molecule_pref_name", ""),
            "activity_type":      act.get("standard_type",     ""),
            "activity_value":     act.get("standard_value",    ""),
            "activity_units":     act.get("standard_units",    ""),
            "activity_relation":  act.get("standard_relation", ""),
            "assay_type":         act.get("assay_type",        ""),
            "document_year":      act.get("document_year",     ""),
            "chembl_url":         f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_id}/",
        })

    return {
        "gene":            gene_symbol,
        "target_chembl_id":target_id,
        "target_name":     target_name,
        "total_activities":act_data.get("page_meta", {}).get("total_count", len(drugs)),
        "drugs":           drugs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL — compound details
# ─────────────────────────────────────────────────────────────────────────────

@cached("drug_target")
@rate_limited("chembl")
@with_retry(max_attempts=3)
async def get_compound_info(chembl_id: str) -> dict[str, Any]:
    """
    Retrieve detailed information about a drug/compound.

    Args:
        chembl_id: ChEMBL compound ID (e.g. 'CHEMBL25' for aspirin).

    Returns:
        SMILES, InChI, molecular properties (Lipinski Ro5, QED score),
        drug approval phase, and therapeutic indications.
    """
    chembl_id = BioValidator.validate_chembl_id(chembl_id)
    client    = await get_http_client()

    resp = await client.get(
        f"{CHEMBL_BASE}/molecule/{chembl_id}.json",
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:
        return {"error": f"Compound '{chembl_id}' not found in ChEMBL."}
    resp.raise_for_status()

    d       = resp.json()
    props   = d.get("molecule_properties") or {}
    structs = d.get("molecule_structures")  or {}
    drug_ind = d.get("drug_indications")   or []

    return {
        "chembl_id":        chembl_id,
        "name":             d.get("pref_name", ""),
        "molecule_type":    d.get("molecule_type", ""),
        "max_phase":        d.get("max_phase", 0),
        "drug_approved":    d.get("max_phase", 0) == 4,
        "smiles":           structs.get("canonical_smiles", ""),
        "inchi":            structs.get("standard_inchi",   ""),
        "inchi_key":        structs.get("standard_inchi_key",""),
        "molecular_formula":props.get("full_molformula",   ""),
        "molecular_weight": props.get("full_mwt",          ""),
        "alogp":            props.get("alogp",             ""),
        "hbd":              props.get("hbd",               ""),
        "hba":              props.get("hba",               ""),
        "psa":              props.get("psa",               ""),
        "rotatable_bonds":  props.get("rtb",               ""),
        "ro5_violations":   props.get("num_ro5_violations",""),
        "qed_weighted":     props.get("qed_weighted",      ""),
        "oral":             props.get("oral",              False),
        "drug_indications": [
            {
                "indication": ind.get("indication", ""),
                "max_phase":  ind.get("max_phase_for_ind", 0),
                "mesh_id":    ind.get("mesh_id", ""),
            }
            for ind in drug_ind[:10]
        ],
        "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Open Targets — gene-disease associations
# ─────────────────────────────────────────────────────────────────────────────

_OT_SEARCH_Q = """
query SearchGene($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits { id }
  }
}
"""

_OT_ASSOC_Q = """
query GeneDisease($eid: String!, $size: Int!) {
  target(ensemblId: $eid) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(orderByScore: "score", page: {index: 0, size: $size}) {
      count
      rows {
        disease { id name description therapeuticAreas { id name } }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""


@cached("drug_target")
@rate_limited("opentargets")
@with_retry(max_attempts=3)
async def get_gene_disease_associations(
    gene_symbol: str,
    max_results: int = 15,
) -> dict[str, Any]:
    """
    Get gene-disease associations from Open Targets Platform.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'BRCA1', 'KRAS').
        max_results: Associations to return (1–50).

    Returns:
        { gene, ensembl_id, approved_name, total_associations,
          associations: [{ disease_name, score, evidence_by_datatype }] }

    Evidence score datatypes: genetic_association · somatic_mutation ·
        known_drug · animal_model · affected_pathway · literature
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    # 1 — resolve symbol → Ensembl ID
    sr = await client.post(
        OPENTARGETS_GQL,
        json={"query": _OT_SEARCH_Q, "variables": {"q": gene_symbol}},
        headers={"Content-Type": "application/json"},
    )
    sr.raise_for_status()
    hits = sr.json().get("data", {}).get("search", {}).get("hits", [])
    if not hits:
        return {"gene": gene_symbol, "associations": [],
                "error": f"'{gene_symbol}' not found in Open Targets."}

    ensembl_id = hits[0]["id"]

    # 2 — fetch associations
    ar = await client.post(
        OPENTARGETS_GQL,
        json={"query": _OT_ASSOC_Q,
              "variables": {"eid": ensembl_id, "size": max_results}},
        headers={"Content-Type": "application/json"},
    )
    ar.raise_for_status()
    target = ar.json().get("data", {}).get("target", {})
    if not target:
        return {"gene": gene_symbol, "associations": []}

    rows         = target.get("associatedDiseases", {}).get("rows", [])
    total        = target.get("associatedDiseases", {}).get("count", len(rows))
    associations = [
        {
            "disease_id":            row["disease"].get("id",   ""),
            "disease_name":          row["disease"].get("name", ""),
            "description":           (row["disease"].get("description") or "")[:300],
            "therapeutic_areas":     [ta["name"] for ta in row["disease"].get("therapeuticAreas", [])],
            "overall_score":         round(row.get("score", 0), 4),
            "evidence_by_datatype":  {s["id"]: round(s["score"], 3) for s in row.get("datatypeScores", [])},
            "url":                   (
                f"https://platform.opentargets.org/evidence/{ensembl_id}/"
                f"{row['disease'].get('id','')}"
            ),
        }
        for row in rows
    ]

    return {
        "gene":                 gene_symbol,
        "ensembl_id":           ensembl_id,
        "approved_name":        target.get("approvedName", ""),
        "total_associations":   total,
        "associations":         associations,
    }
