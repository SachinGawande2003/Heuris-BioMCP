# 🧬 Heuris-Heuris-BioMCP — Bioinformatics Model Context Protocol Server

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)
[![Tools: 52](https://img.shields.io/badge/Tools-52-blue.svg)](#tools-52-total)
[![Databases: 20+](https://img.shields.io/badge/Databases-20+-purple.svg)](#databases--ai-models)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://heuris-biomcp.onrender.com)

**The most comprehensive Model Context Protocol server for life sciences v2.**  
Connect Claude to every major biological database and state-of-the-art AI models — no API wrappers, no copy-pasting, just science.

[🚀 Quick Start](#quick-start) • [🔧 Tools](#tools-52-total) • [📊 Databases](#databases--ai-models) • [💡 Examples](#usage-examples) • [🤝 Contributing](#contributing)

</div>

---

## Live Demo

Try Heuris-BioMCP without installing — connect to our live server:

```
https://heuris-biomcp.onrender.com/sse
```

### Connect to Live Server

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "biomcp": {
      "url": "https://heuris-biomcp.onrender.com/sse"
    }
  }
}
```

> **Note**: The live demo supports all features. For local development or custom deployments, see [Local Installation](#installation).

---

## See Heuris-BioMCP in Action

![Heuris-BioMCP Demo](assets/biomcp-demo.gif)

### Quick Demo Video

Watch how to connect Heuris-BioMCP to Claude Desktop:

<a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID">
  <img src="assets/youtube-thumbnail.png" alt="Watch Demo" width="400"/>
</a>

> **Tip**: Coming soon - video walkthrough of connecting Heuris-BioMCP and running your first query!

---

## What is Heuris-BioMCP?

Heuris-BioMCP bridges Claude and the world's life sciences databases through the [Model Context Protocol](https://modelcontextprotocol.io). Ask Claude to **search PubMed**, **predict protein structure** with AI, **generate DNA sequences**, **find drug targets**, **query clinical trials**, **analyze single-cell data**, **verify biological claims**, or **generate research hypotheses** — all in natural language, all in real time.

```
You → "What drugs target EGFR and what clinical trials are recruiting?"
Claude + Heuris-BioMCP → Queries ChEMBL + ClinicalTrials.gov simultaneously → Structured answer
```

---

## What's New in v2

- **Extended Databases**: 7 new database integrations (OMIM, STRING, GTEx, cBioPortal, GWAS Catalog, DisGeNET, PharmGKB)
- **Biological Claim Verification**: Verify claims against multiple databases with evidence grading
- **Conflict Detection**: Identify inconsistencies across databases
- **Experimental Design**: Generate protocols, suggest cell lines, calculate statistical power
- **Session Knowledge Graph**: Auto-built entity graph from all tool calls
- **Entity Resolution**: Canonical IDs across HGNC/UniProt/Ensembl/NCBI
- **Adaptive Query Planner**: Dependency-aware parallel execution DAG
- **Reproducibility Export**: Full provenance + citations + reproducible script generation

---

## Tools (42 total)

### 📚 Literature & NCBI
| Tool | Description |
|------|-------------|
| `search_pubmed` | Full PubMed search with MeSH, Boolean, field tags. Returns abstracts, authors, DOIs |
| `get_gene_info` | NCBI Gene: symbol, location, aliases, RefSeq IDs, functional summary |
| `run_blast` | NCBI BLAST alignment (blastp/blastn/blastx/tblastn) — async, non-blocking |

### 🧪 Proteins & Structures
| Tool | Description |
|------|-------------|
| `get_protein_info` | UniProt Swiss-Prot: function, domains, PTMs, GO terms, disease links |
| `search_proteins` | Search UniProt by gene, function, or disease with species filtering |
| `get_alphafold_structure` | AlphaFold DB: pLDDT confidence scores, PDB/mmCIF download URLs, PAE data |
| `search_pdb_structures` | RCSB PDB: experimental structures with resolution, method, deposition date |

### 🔬 Pathways
| Tool | Description |
|------|-------------|
| `search_pathways` | KEGG pathway search with organism-specific viewer links |
| `get_pathway_genes` | All genes in a KEGG pathway with descriptions |
| `get_reactome_pathways` | Reactome pathways for a gene with hierarchy and diagram links |

### 💊 Drug Discovery
| Tool | Description |
|------|-------------|
| `get_drug_targets` | ChEMBL: drugs/compounds targeting a gene with IC50, Ki, approval status |
| `get_compound_info` | ChEMBL compound details: SMILES, ADMET, Lipinski Ro5, QED, indications |
| `get_gene_disease_associations` | Open Targets: gene-disease evidence across genetics, drugs, and pathways |

### 🧬 Genomics & Expression
| Tool | Description |
|------|-------------|
| `get_gene_variants` | Ensembl variants: SNPs, indels, consequence types, clinical significance |
| `search_gene_expression` | NCBI GEO datasets for a gene with organism, platform, sample counts |
| `search_scrna_datasets` | Human Cell Atlas single-cell RNA-seq by tissue and technology |

### 🏥 Clinical
| Tool | Description |
|------|-------------|
| `search_clinical_trials` | ClinicalTrials.gov: trials with status, phase, interventions, eligibility |
| `get_trial_details` | Full trial protocol: arms, outcomes, contacts |

### 🤖 AI-Powered (NVIDIA NIM)
| Tool | Description |
|------|-------------|
| `predict_structure_boltz2` | MIT Boltz-2: Protein/DNA/RNA/ligand structure prediction + binding affinity (FEP accuracy, 1000x faster) |
| `generate_dna_evo2` | Arc Evo2-40B: Generate DNA sequences with 40B parameter genomic foundation model |
| `score_sequence_evo2` | Compare wildtype vs variant DNA sequences for variant effect prediction |
| `design_protein_ligand` | Full drug-discovery pipeline: UniProt → Boltz-2 structure + affinity in one call |

### 🌐 Integrated & Advanced
| Tool | Description |
|------|-------------|
| `multi_omics_gene_report` | **Flagship**: 7+ databases queried in parallel → one integrated gene report |
| `query_neuroimaging_datasets` | OpenNeuro + NeuroVault: fMRI/EEG/MEG datasets with acquisition metadata |
| `generate_research_hypothesis` | Literature mining → data-driven testable hypotheses with supporting evidence |

### 🗄️ Extended Databases (v2)
| Tool | Description |
|------|-------------|
| `get_omim_gene_diseases` | OMIM: gene-disease relationships, inheritance patterns, phenotypes |
| `get_string_interactions` | STRING: protein-protein interactions with confidence scores |
| `get_gtex_expression` | GTEx: gene expression by tissue with TPM/FPKM values |
| `search_cbio_mutations` | cBioPortal: cancer genomics mutations, copy number alterations |
| `search_gwas_catalog` | GWAS Catalog: trait-associated SNPs with p-values and effect sizes |
| `get_disgenet_associations` | DisGeNET: gene-disease associations with evidence scores |
| `get_pharmgkb_variants` | PharmGKB: pharmacogenomic variants and drug responses |

### ✅ Verification & Conflict Detection (v2)
| Tool | Description |
|------|-------------|
| `verify_biological_claim` | Verify claims against multiple databases with graded evidence (strong/moderate/weak/contradicted) |
| `detect_database_conflicts` | Identify inconsistencies across databases with conflict resolution suggestions |

### 🧬 Experimental Design (v2)
| Tool | Description |
|------|-------------|
| `generate_experimental_protocol` | Generate step-by-step experimental protocols with controls and timelines |
| `suggest_cell_lines` | Recommend appropriate cell lines based on research goals |
| `estimate_statistical_power` | Calculate statistical power and sample size requirements |

### 🧠 Session Intelligence (v2)
| Tool | Description |
|------|-------------|
| `resolve_entity` | Canonical cross-database entity resolution (HGNC/UniProt/Ensembl/NCBI) |
| `get_session_knowledge_graph` | Live entity graph auto-built from all tool calls in session |
| `find_biological_connections` | Discover cross-database connections between entities |
| `export_research_session` | Full provenance + citations + reproducible Python script |
| `plan_and_execute_research` | DAG-based adaptive research workflow planner |

---

## Databases & AI Models

| Source | Domain | URL |
|--------|--------|-----|
| **Traditional Databases** |
| NCBI PubMed | Literature | https://pubmed.ncbi.nlm.nih.gov |
| NCBI Gene | Genomics | https://www.ncbi.nlm.nih.gov/gene |
| NCBI BLAST | Sequence Alignment | https://blast.ncbi.nlm.nih.gov |
| NCBI GEO | Gene Expression | https://www.ncbi.nlm.nih.gov/geo |
| UniProt Swiss-Prot | Proteomics | https://www.uniprot.org |
| AlphaFold DB | Protein Structure | https://alphafold.ebi.ac.uk |
| RCSB PDB | Protein Structure | https://www.rcsb.org |
| KEGG | Pathways | https://www.kegg.jp |
| Reactome | Pathways | https://reactome.org |
| ChEMBL | Drug Discovery | https://www.ebi.ac.uk/chembl |
| Open Targets | Gene-Disease | https://platform.opentargets.org |
| Ensembl | Genomics | https://www.ensembl.org |
| ClinicalTrials.gov | Clinical | https://clinicaltrials.gov |
| Human Cell Atlas | Single-Cell | https://data.humancellatlas.org |
| OpenNeuro | Neuroimaging | https://openneuro.org |
| NeuroVault | Neuroimaging | https://neurovault.org |
| **v2 Extended Databases** |
| OMIM | Genetic Diseases | https://www.omim.org |
| STRING | Protein Interactions | https://string-db.org |
| GTEx | Expression Atlas | https://gtexportal.org |
| cBioPortal | Cancer Genomics | https://www.cbioportal.org |
| GWAS Catalog | Trait Associations | https://www.ebi.ac.uk/gwas |
| DisGeNET | Disease-Gene | https://www.disgenet.org |
| PharmGKB | Pharmacogenomics | https://www.pharmgkb.org |
| **AI Models (NVIDIA NIM)** |
| MIT Boltz-2 | Structure Prediction | https://build.nvidia.com/mit/boltz2 |
| Arc Evo2-40B | DNA Generation | https://build.nvidia.com/arc/evo2-40b |

---

## Quick Start

### Option 1: Use Live Demo (No Installation)

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "biomcp": {
      "url": "https://heuris-biomcp.onrender.com/sse"
    }
  }
}
```

### Option 2: Deploy Your Own

Deploy to Render with one click:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SachinGawande2003/Heuris-BioMCP)

Or manually:
1. Fork this repository
2. Create a new Web Service on [Render](https://render.com)
3. Connect your fork
4. Set build command: `pip install -r requirements.txt && pip install -e .`
5. Set start command: `BIOMCP_TRANSPORT=http BIOMCP_HTTP_PORT=$PORT python -m biomcp`

### Option 3: Local Installation

#### Prerequisites
- Python 3.11+
- Claude Desktop or any MCP-compatible client
- (Optional) NCBI API key for higher rate limits
- (Optional) NVIDIA API keys for AI tools

#### Installation

```bash
# Clone the repository
git clone https://github.com/SachinGawande2003/biomcp.git
cd biomcp

# Install (standard)
pip install -e .

# Install with neuroimaging support
pip install -e ".[neuroimaging]"

# Install with dev dependencies (recommended)
pip install -e ".[dev]"
```

#### Configure Claude Desktop

**For Local STDIO Mode (default):**

```json
{
  "mcpServers": {
    "biomcp": {
      "command": "biomcp",
      "env": {
        "NCBI_API_KEY": "your_ncbi_api_key_here"
      }
    }
  }
}
```

**For Remote HTTP Mode (using live demo or your own deployed server):**

```json
{
  "mcpServers": {
    "biomcp": {
      "url": "https://heuris-biomcp.onrender.com/sse"
    }
  }
}
```

> **💡 Tip**: Get a free [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) to increase rate limits from 3 to 10 requests/second.
>
> **🚀 New**: Get free NVIDIA API keys for AI tools at [build.nvidia.com/mit/boltz2](https://build.nvidia.com/mit/boltz2) and [build.nvidia.com/arc/evo2-40b](https://build.nvidia.com/arc/evo2-40b).

### Restart Claude Desktop and test:

```
"Search PubMed for recent papers on CAR-T cell therapy in B-cell lymphoma"
"Get the AlphaFold structure for TP53 and tell me about the confidence scores"
"What drugs are approved that target EGFR?"
"Generate a multi-omics report for KRAS"
"Predict the structure of EGFR with ligand CC1=CC=CC=C1 and compute binding affinity"
"Generate a DNA sequence starting with ATGGCG..."
```

---

## Usage Examples

### Literature Mining
```
"Search PubMed for BRCA1 CRISPR correction methods published in the last 2 years"
"Find review articles about PD-1/PD-L1 immune checkpoint inhibitors"
```

### Protein Analysis
```
"Get UniProt info for human TP53 (P04637) including its domains and disease associations"
"Search for AlphaFold structures for insulin receptor"
"Find all PDB crystal structures of BRAF kinase domain resolved below 2.5 Ångström"
```

### Drug Discovery
```
"What are the top ChEMBL compounds targeting KRAS G12C mutation?"
"Get compound info for imatinib (CHEMBL941)"
"Show me gene-disease associations for BRCA1 with evidence scores"
```

### AI-Powered Structure Prediction
```
"Predict the 3D structure of insulin (sequence: ...) with ligand CCO"
"Compute binding affinity between EGFR and gefitinib (SMILES: ...)"
"Get structure prediction for a protein-protein complex"
```

### AI-Powered DNA Generation
```
"Generate a 200bp promoter sequence starting with ATG"
"Compare wildtype vs variant DNA sequence for TP53 mutation"
"Generate regulatory element for gene expression"
```

### Multi-Omics Report (Flagship)
```
"Generate a complete multi-omics report for EGFR"
```
This single command queries **7 databases in parallel** and returns:
- Genomic location and gene summary (NCBI Gene)
- Recent publications (PubMed)  
- Protein function and structure (UniProt + AlphaFold)
- Biological pathways (Reactome)
- Drug targets and clinical compounds (ChEMBL)
- Disease associations with scores (Open Targets)
- Expression datasets (GEO)
- Active clinical trials (ClinicalTrials.gov)

### v2 Extended Databases
```
"Get OMIM diseases associated with TP53"
"Show STRING protein interactions for EGFR"
"Get GTEx expression data for BRCA1 across tissues"
"Find mutations in TP53 from cBioPortal"
"Search GWAS for diabetes-associated SNPs"
```

### v2 Verification
```
"Verify the claim that TP53 is a tumor suppressor gene"
"Detect conflicts between OMIM and DisGeNET for BRCA1"
```

### v2 Experimental Design
```
"Generate an experimental protocol for CRISPR knockout of BRCA1"
"What cell lines should I use to study KRAS mutations?"
"Calculate sample size for detecting 2-fold change with p<0.05"
```

### v2 Session Intelligence
```
"What's the knowledge graph from our conversation so far?"
"Find biological connections between TP53 and EGFR"
"Export our research session as a reproducible script"
"Plan and execute a research workflow for PD-1 drug targets"
```

---

## Architecture

```
biomcp/
├── src/biomcp/
│   ├── server.py              # MCP server — tool registry & dispatcher
│   ├── tools/
│   │   ├── ncbi.py            # PubMed, Gene, BLAST
│   │   ├── proteins.py        # UniProt, AlphaFold, PDB
│   │   ├── pathways.py        # KEGG, Reactome, ChEMBL, Open Targets
│   │   ├── advanced.py        # ClinicalTrials, GEO, scRNA, Ensembl,
│   │   │                      # Multi-Omics, Neuroimaging, Hypothesis
│   │   ├── nvidia_nim.py      # Boltz-2, Evo2-40B AI models
│   │   ├── databases.py       # v2: OMIM, STRING, GTEx, cBioPortal,
│   │   │                      #      GWAS, DisGeNET, PharmGKB
│   │   ├── verify.py          # v2: Claim verification, conflict detection
│   │   └── protocol_generator.py  # v2: Experimental design tools
│   ├── core/
│   │   ├── entity_resolver.py # v2: Cross-database entity resolution
│   │   ├── knowledge_graph.py # v2: Session knowledge graph
│   │   └── query_planner.py   # v2: Adaptive query planner
│   └── utils/
│       └── __init__.py        # Rate limiter, cache, validators, HTTP client
├── tests/
├── pyproject.toml
└── README.md
```

**Key design decisions:**
- **Async-first**: All API calls are fully async with `httpx`, never blocking
- **Rate limiting**: Token-bucket limiter per service respects each API's limits
- **Smart caching**: TTL-based per-namespace cache (1h literature, 7d structures)  
- **Retry logic**: Exponential backoff via `tenacity` for transient failures
- **Validation**: Input validation before any network call — never wastes API quota

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NCBI_API_KEY` | NCBI API key (increases rate limit to 10/s) | None (3/s) |
| `NVIDIA_BOLTZ2_API_KEY` | NVIDIA API key for Boltz-2 structure prediction | None |
| `NVIDIA_EVO2_API_KEY` | NVIDIA API key for Evo2-40B DNA generation | None |
| `BIOMCP_LOG_LEVEL` | Log level: DEBUG/INFO/WARNING/ERROR | INFO |

Get free API keys:
- **NCBI**: https://www.ncbi.nlm.nih.gov/account/
- **NVIDIA Boltz-2**: https://build.nvidia.com/mit/boltz2
- **NVIDIA Evo2-40B**: https://build.nvidia.com/arc/evo2-40b

---

## Contributing

Contributions are welcome! Whether it's adding a new database, fixing a bug, improving documentation, or integrating new AI models.

```bash
# Development setup
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=biomcp

# Lint + type check
ruff check src/
mypy src/
```

### Ideas for contributions
- Add more pathway databases (Wikipathways, PathCards)
- Integrate COSMIC for somatic mutations
- Add protein complex data (CORUM)
- Implement batch query support for high-throughput analysis
- Add Jupyter notebook examples
- Improve conflict resolution algorithms

---

## Citation

If you use Heuris-BioMCP in your research, please cite:

```bibtex
@software{biomcp2025,
  title   = {Heuris-BioMCP v2: A Comprehensive MCP Server for Bioinformatics, AI Models, and Life Sciences},
  year    = {2025},
  url     = {https://github.com/SachinGawande2003/biomcp},
  license = {MIT}
}
```

---

## License

MIT License — free for academic and commercial use.

---

<div align="center">
Built for researchers, by researchers. 🔬<br>
<b>Star ⭐ this repo if Heuris-BioMCP helps your science!</b>
</div>
