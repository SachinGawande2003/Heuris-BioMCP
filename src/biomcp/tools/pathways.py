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

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

KEGG_BASE         = "https://rest.kegg.jp"
REACTOME_BASE     = "https://reactome.org/ContentService"
REACTOME_ANALYSIS = "https://reactome.org/AnalysisService"
CHEMBL_BASE       = "https://www.ebi.ac.uk/chembl/api/data"
OPENTARGETS_GQL   = "https://api.platform.opentargets.org/api/v4/graphql"

KEGG_ORGANISM_TO_NCBI = {
    "cel": "caenorhabditis elegans",
    "dme": "drosophila melanogaster",
    "dre": "danio rerio",
    "hsa": "homo sapiens",
    "mmu": "mus musculus",
    "rno": "rattus norvegicus",
    "sce": "saccharomyces cerevisiae",
}


def _kegg_organism_name(organism: str) -> str:
    return KEGG_ORGANISM_TO_NCBI.get(organism.lower(), organism.replace("_", " "))


def _parse_kegg_flat_records(raw_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in raw_text.split("///"):
        block = block.strip()
        if not block:
            continue

        fields: dict[str, str] = {}
        current_key = ""
        for line in block.splitlines():
            key = line[:12].strip()
            value = line[12:].strip()
            if key:
                current_key = key
                fields[key] = f"{fields.get(key, '')} {value}".strip()
            elif current_key and value:
                fields[current_key] = f"{fields[current_key]} {value}".strip()

        entry_id = fields.get("ENTRY", "").split()[0]
        if not entry_id:
            continue

        description = fields.get("NAME", "").rstrip(";").strip()
        records.append({
            "pathway_id": entry_id,
            "organism_id": entry_id,
            "description": description,
            "summary": fields.get("DESCRIPTION", ""),
            "category": fields.get("CLASS", ""),
            "viewer_url": f"https://www.kegg.jp/pathway/{entry_id}",
            "image_url": f"https://www.kegg.jp/kegg/pathway/{entry_id}/{entry_id}.png",
        })

    return records


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


@cached("kegg")
@rate_limited("kegg")
@with_retry(max_attempts=3)
async def get_kegg_gene_pathways(gene_symbol: str, organism: str = "hsa") -> dict[str, Any]:
    """
    Resolve a gene symbol to KEGG pathways that contain that gene.

    This uses KEGG membership links rather than keyword search:
      gene symbol -> NCBI Gene ID -> KEGG gene ID -> linked pathways
    """
    from biomcp.tools.ncbi import get_gene_info

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    organism = organism.strip().lower()
    client = await get_http_client()

    gene_info = await get_gene_info(gene_symbol, organism=_kegg_organism_name(organism))
    ncbi_gene_id = str(gene_info.get("gene_id", "")).strip()
    if not ncbi_gene_id:
        return {
            "gene": gene_symbol,
            "organism": organism,
            "kegg_gene_ids": [],
            "pathways": [],
            "total": 0,
            "error": gene_info.get(
                "error",
                f"Could not resolve an NCBI Gene ID for '{gene_symbol}' in '{organism}'.",
            ),
        }

    conv_resp = await client.get(
        f"{KEGG_BASE}/conv/{organism}/ncbi-geneid:{ncbi_gene_id}",
        headers={"Accept": "text/plain"},
    )
    if conv_resp.status_code == 404:
        return {
            "gene": gene_symbol,
            "organism": organism,
            "ncbi_gene_id": ncbi_gene_id,
            "kegg_gene_ids": [],
            "pathways": [],
            "total": 0,
            "note": f"No KEGG gene mapping found for '{gene_symbol}' in organism '{organism}'.",
        }
    conv_resp.raise_for_status()

    kegg_gene_ids: list[str] = []
    for line in conv_resp.text.strip().splitlines():
        if "\t" not in line:
            continue
        _, kegg_gene_id = line.split("\t", 1)
        kegg_gene_id = kegg_gene_id.strip()
        if kegg_gene_id and kegg_gene_id not in kegg_gene_ids:
            kegg_gene_ids.append(kegg_gene_id)

    if not kegg_gene_ids:
        return {
            "gene": gene_symbol,
            "organism": organism,
            "ncbi_gene_id": ncbi_gene_id,
            "kegg_gene_ids": [],
            "pathways": [],
            "total": 0,
            "note": f"No KEGG gene mapping found for '{gene_symbol}' in organism '{organism}'.",
        }

    pathway_ids: list[str] = []
    for kegg_gene_id in kegg_gene_ids:
        link_resp = await client.get(
            f"{KEGG_BASE}/link/pathway/{kegg_gene_id}",
            headers={"Accept": "text/plain"},
        )
        if link_resp.status_code == 404:
            continue
        link_resp.raise_for_status()
        for line in link_resp.text.strip().splitlines():
            if "\t" not in line:
                continue
            _, path_ref = line.split("\t", 1)
            pathway_id = path_ref.replace("path:", "").strip()
            if pathway_id and pathway_id not in pathway_ids:
                pathway_ids.append(pathway_id)

    if not pathway_ids:
        return {
            "gene": gene_symbol,
            "organism": organism,
            "ncbi_gene_id": ncbi_gene_id,
            "kegg_gene_ids": kegg_gene_ids,
            "pathways": [],
            "total": 0,
        }

    pathway_lookup: dict[str, dict[str, Any]] = {}
    for i in range(0, len(pathway_ids), 10):
        batch = pathway_ids[i:i + 10]
        detail_resp = await client.get(
            f"{KEGG_BASE}/get/{'+'.join(batch)}",
            headers={"Accept": "text/plain"},
        )
        if detail_resp.status_code != 200:
            continue
        for record in _parse_kegg_flat_records(detail_resp.text):
            pathway_lookup[record["pathway_id"]] = record

    pathways = [
        pathway_lookup.get(
            pathway_id,
            {
                "pathway_id": pathway_id,
                "organism_id": pathway_id,
                "description": pathway_id,
                "summary": "",
                "category": "",
                "viewer_url": f"https://www.kegg.jp/pathway/{pathway_id}",
                "image_url": f"https://www.kegg.jp/kegg/pathway/{pathway_id}/{pathway_id}.png",
            },
        )
        for pathway_id in pathway_ids
    ]

    return {
        "gene": gene_symbol,
        "organism": organism,
        "ncbi_gene_id": ncbi_gene_id,
        "kegg_gene_ids": kegg_gene_ids,
        "total": len(pathways),
        "pathways": pathways,
    }


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

    analysis_resp = await client.post(
        f"{REACTOME_ANALYSIS}/identifiers/projection",
        content=gene_symbol,
        headers={"Content-Type": "text/plain", "Accept": "application/json"},
    )
    if analysis_resp.status_code == 404:
        return {
            "gene": gene_symbol,
            "species_taxid": species,
            "total": 0,
            "pathways": [],
            "note": f"'{gene_symbol}' not found in Reactome.",
        }
    analysis_resp.raise_for_status()

    payload = analysis_resp.json()
    pathways: list[dict[str, Any]] = []
    for entry in payload.get("pathways", []):
        stid = entry.get("stId", "")
        species_info = entry.get("species") or {}
        species_taxid = str(species_info.get("taxId", ""))
        if not stid or (species and species_taxid and species_taxid != species):
            continue

        entity_stats = entry.get("entities") or {}
        pathways.append({
            "reactome_id": stid,
            "name": entry.get("name", ""),
            "type": "DiseasePathway" if entry.get("inDisease") else "Pathway",
            "species": species_info.get("name", ""),
            "url": f"https://reactome.org/content/detail/{stid}",
            "diagram_url": f"https://reactome.org/PathwayBrowser/#/{stid}",
            "found_entities": entity_stats.get("found", 0),
            "total_entities": entity_stats.get("total", 0),
            "p_value": entity_stats.get("pValue"),
            "fdr": entity_stats.get("fdr"),
            "in_disease": bool(entry.get("inDisease")),
        })

    return {
        "gene": gene_symbol,
        "species_taxid": species,
        "total": len(pathways),
        "pathways": pathways,
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
