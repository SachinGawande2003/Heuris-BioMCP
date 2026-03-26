"""
BioMCP — NVIDIA NIM Tools
==========================
State-of-the-art AI models via NVIDIA NIM API:

  predict_structure_boltz2   — MIT Boltz-2: Biomolecular structure prediction
                               + binding affinity (approaches FEP accuracy,
                               1000x faster). Proteins · DNA · RNA · Ligands.

  generate_dna_evo2          — Arc Evo2-40B: 40-billion parameter genomic
                               foundation model. Generate / score DNA sequences
                               with single-nucleotide sensitivity.

  score_sequence_evo2        — Score a DNA sequence for likelihood under Evo2
                               (uses logits reporting).

  design_protein_ligand      — Full drug-discovery pipeline: UniProt lookup →
                               Boltz-2 structure + affinity prediction, all in
                               one tool call.

API Base:
  https://health.api.nvidia.com/v1/biology/

Requirements (two separate keys — each model has its own):
  NVIDIA_BOLTZ2_API_KEY  — for MIT Boltz-2 structure + affinity prediction
                           Get free key at: https://build.nvidia.com/mit/boltz2

  NVIDIA_EVO2_API_KEY    — for Arc Evo2-40B DNA generation + scoring
                           Get free key at: https://build.nvidia.com/arc/evo2-40b
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    format_error,
    format_success,
    get_http_client,
    rate_limited,
    with_retry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_NIM_BASE      = "https://health.api.nvidia.com/v1/biology"
_BOLTZ2_URL    = f"{_NIM_BASE}/mit/boltz2/predict"
_EVO2_URL      = f"{_NIM_BASE}/arc/evo2-40b/generate"

# ── Separate API keys for each model ─────────────────────────────────────────
# Boltz-2 key  → https://build.nvidia.com/mit/boltz2   → Get API Key
# Evo2-40B key → https://build.nvidia.com/arc/evo2-40b → Get API Key

BOLTZ2_API_KEY: str = (
    os.getenv("NVIDIA_BOLTZ2_API_KEY")
    or os.getenv("NVIDIA_NIM_API_KEY")   # fallback: single key for both
    or ""
)

EVO2_API_KEY: str = (
    os.getenv("NVIDIA_EVO2_API_KEY")
    or os.getenv("NVIDIA_NIM_API_KEY")   # fallback: single key for both
    or ""
)


def _boltz2_headers() -> dict[str, str]:
    """Auth headers for Boltz-2 API calls."""
    if not BOLTZ2_API_KEY:
        raise EnvironmentError(
            "Boltz-2 API key not set. "
            "Add NVIDIA_BOLTZ2_API_KEY=nvapi-... to your .env file.\n"
            "Get a free key at: https://build.nvidia.com/mit/boltz2 → 'Get API Key'"
        )
    return {
        "Authorization": f"Bearer {BOLTZ2_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def _evo2_headers() -> dict[str, str]:
    """Auth headers for Evo2-40B API calls."""
    if not EVO2_API_KEY:
        raise EnvironmentError(
            "Evo2-40B API key not set. "
            "Add NVIDIA_EVO2_API_KEY=nvapi-... to your .env file.\n"
            "Get a free key at: https://build.nvidia.com/arc/evo2-40b → 'Get API Key'"
        )
    return {
        "Authorization": f"Bearer {EVO2_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Boltz-2 — Structure Prediction + Binding Affinity
# ─────────────────────────────────────────────────────────────────────────────

@rate_limited("default")
@with_retry(max_attempts=3, min_wait=2.0, max_wait=15.0)
async def predict_structure_boltz2(
    protein_sequences: list[str],
    ligand_smiles: list[str] | None = None,
    dna_sequences: list[str] | None = None,
    rna_sequences: list[str] | None = None,
    predict_affinity: bool = False,
    method_conditioning: str | None = None,
    pocket_residues: list[dict] | None = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
) -> dict[str, Any]:
    """
    Predict biomolecular complex structure using MIT Boltz-2 via NVIDIA NIM.

    Boltz-2 is the first deep learning model to approach FEP accuracy in
    binding affinity prediction while being ~1000x more computationally
    efficient. Supports proteins, DNA, RNA, small molecule ligands, and
    covalent modifications.

    Args:
        protein_sequences:   List of protein amino acid sequences (FASTA format,
                             max 4096 residues per chain, max 12 chains total).
        ligand_smiles:       List of ligand SMILES strings (max 20 ligands).
                             E.g. ['CC1=CC=CC=C1', 'c1ccccc1'] for toluene + benzene.
        dna_sequences:       Optional DNA chain sequences (5'→3').
        rna_sequences:       Optional RNA chain sequences (5'→3').
        predict_affinity:    If True, compute binding affinity between ligands
                             and the rest of the complex (requires ≥1 ligand +
                             ≥1 protein). Returns IC50-like affinity score.
        method_conditioning: Steer output structure style:
                             'x-ray' | 'nmr' | 'md' | None (default, unbiased).
        pocket_residues:     List of pocket/constraint dicts for guided docking.
                             E.g. [{'chain': 'A', 'residue': 42}]
        recycling_steps:     Number of recycling iterations (1–10). Default 3.
        sampling_steps:      Diffusion sampling steps (50–500). Default 200.
        diffusion_samples:   Number of structural hypotheses (1–5). Default 1.

    Returns:
        {
          status, job_id, model, input_summary,
          structure: {
            mmcif_data,        # Full mmCIF structure string
            format             # 'mmcif'
          },
          scores: {
            iptm,              # Interface pTM (complex quality, 0–1, higher better)
            ptm,               # Overall pTM (chain quality, 0–1)
            confidence,        # Overall confidence score
            per_chain_ptm,     # Per-chain breakdown
          },
          affinity: {          # Only if predict_affinity=True
            affinity_pred,     # Predicted log µM affinity (IC50-like)
            affinity_probability_binary,  # P(binding) 0–1
          },
          runtime_metrics,
          visualization_note
        }
    """
    if not protein_sequences:
        raise ValueError("At least one protein sequence is required.")

    # Validate sequences
    validated_proteins = [
        BioValidator.validate_sequence(seq, "protein")
        for seq in protein_sequences
    ]

    if ligand_smiles is None:
        ligand_smiles = []
    if dna_sequences is None:
        dna_sequences = []
    if rna_sequences is None:
        rna_sequences = []

    recycling_steps  = BioValidator.clamp_int(recycling_steps,  1, 10,  "recycling_steps")
    sampling_steps   = BioValidator.clamp_int(sampling_steps,   50, 500, "sampling_steps")
    diffusion_samples= BioValidator.clamp_int(diffusion_samples, 1, 5,   "diffusion_samples")

    if predict_affinity and not ligand_smiles:
        raise ValueError(
            "predict_affinity=True requires at least one ligand_smiles entry."
        )

    if method_conditioning and method_conditioning not in ("x-ray", "nmr", "md"):
        raise ValueError(
            f"method_conditioning must be 'x-ray', 'nmr', 'md', or None. "
            f"Got '{method_conditioning}'."
        )

    # ── Build request payload ─────────────────────────────────────────────────
    sequences: list[dict] = []

    for i, seq in enumerate(validated_proteins):
        sequences.append({
            "protein": {
                "id":       chr(65 + i),   # A, B, C, ...
                "sequence": seq,
            }
        })

    for i, smiles in enumerate(ligand_smiles):
        sequences.append({
            "ligand": {
                "id":     f"LIG{i + 1}",
                "smiles": smiles,
            }
        })

    for i, seq in enumerate(dna_sequences):
        sequences.append({
            "dna": {
                "id":       f"DNA{i + 1}",
                "sequence": BioValidator.validate_sequence(seq, "nucleotide"),
            }
        })

    for i, seq in enumerate(rna_sequences):
        sequences.append({
            "rna": {
                "id":       f"RNA{i + 1}",
                "sequence": BioValidator.validate_sequence(seq, "nucleotide"),
            }
        })

    payload: dict[str, Any] = {
        "sequences":          sequences,
        "predict_affinity":   predict_affinity,
        "recycling_steps":    recycling_steps,
        "sampling_steps":     sampling_steps,
        "diffusion_samples":  diffusion_samples,
        "output_format":      "mmcif",
        "return_runtime_metrics": True,
    }

    if method_conditioning:
        payload["method_conditioning"] = method_conditioning

    if pocket_residues:
        payload["pocket_conditioning"] = pocket_residues

    logger.info(
        f"[Boltz-2] Predicting structure: "
        f"{len(validated_proteins)} protein(s), "
        f"{len(ligand_smiles)} ligand(s), "
        f"affinity={predict_affinity}"
    )

    # ── Call NVIDIA NIM ───────────────────────────────────────────────────────
    client   = await get_http_client()
    headers  = _boltz2_headers()
    t_start  = time.monotonic()

    resp = await client.post(
        _BOLTZ2_URL,
        headers=headers,
        json=payload,
        timeout=300.0,    # Boltz-2 can take up to 5 min for large complexes
    )

    if resp.status_code == 402:
        raise PermissionError(
            "NVIDIA NIM quota exceeded. "
            "Upgrade at build.nvidia.com or wait for quota reset."
        )
    if resp.status_code == 401:
        raise PermissionError(
            "Invalid NVIDIA API key. Check NVIDIA_NIM_API_KEY in your .env file."
        )
    resp.raise_for_status()

    data     = resp.json()
    elapsed  = round(time.monotonic() - t_start, 2)

    # ── Parse response ────────────────────────────────────────────────────────
    # The NIM may return structure as base64-encoded mmcif or direct string
    structure_raw = data.get("mmcif") or data.get("structure") or ""
    if isinstance(structure_raw, bytes):
        structure_raw = structure_raw.decode("utf-8")
    # Sometimes returned as base64
    if structure_raw and not structure_raw.strip().startswith("data_"):
        try:
            structure_raw = base64.b64decode(structure_raw).decode("utf-8")
        except Exception:
            pass

    scores    = data.get("scores", {})
    affinity  = data.get("affinity", {})
    runtime   = data.get("runtime_metrics", {})

    result: dict[str, Any] = {
        "status":        "success",
        "model":         "MIT Boltz-2 via NVIDIA NIM",
        "elapsed_s":     elapsed,
        "input_summary": {
            "protein_chains":    len(validated_proteins),
            "ligands":           len(ligand_smiles),
            "dna_chains":        len(dna_sequences),
            "rna_chains":        len(rna_sequences),
            "predict_affinity":  predict_affinity,
            "method_conditioning": method_conditioning,
            "recycling_steps":   recycling_steps,
            "sampling_steps":    sampling_steps,
        },
        "structure": {
            "mmcif_data": structure_raw[:5_000] if structure_raw else "See full response",
            "mmcif_length_chars": len(structure_raw),
            "format": "mmcif",
        },
        "scores": {
            "iptm":          scores.get("iptm"),
            "ptm":           scores.get("ptm"),
            "confidence":    scores.get("confidence"),
            "per_chain_ptm": scores.get("per_chain_ptm", {}),
            "interpretation": _interpret_boltz_scores(scores),
        },
        "runtime_metrics": runtime,
        "visualization": {
            "note": (
                "Download the mmcif_data and visualize with: "
                "Mol* (https://molstar.org/viewer/), "
                "PyMOL, or UCSF ChimeraX."
            ),
            "online_viewer": "https://molstar.org/viewer/",
        },
    }

    if predict_affinity and affinity:
        aff_val = affinity.get("affinity_pred")
        aff_prob = affinity.get("affinity_probability_binary")
        result["affinity"] = {
            "affinity_pred_log_uM":          aff_val,
            "affinity_probability_binding":  aff_prob,
            "predicted_IC50_uM":             (
                round(10 ** aff_val, 4) if aff_val is not None else None
            ),
            "interpretation": _interpret_affinity(aff_val, aff_prob),
        }
    elif predict_affinity:
        result["affinity"] = {"note": "Affinity data not returned by API for this complex."}

    return result


def _interpret_boltz_scores(scores: dict) -> str:
    """Human-readable confidence interpretation for Boltz-2 scores."""
    conf = scores.get("confidence")
    if conf is None:
        return "No confidence score available."
    if conf >= 0.8:
        return "High confidence — structure prediction is reliable."
    if conf >= 0.6:
        return "Moderate confidence — use with some caution."
    if conf >= 0.4:
        return "Low confidence — validate experimentally."
    return "Very low confidence — likely disordered or insufficient data."


def _interpret_affinity(aff_val: float | None, aff_prob: float | None) -> str:
    """Human-readable affinity interpretation."""
    if aff_val is None:
        return "No affinity prediction available."
    ic50 = 10 ** aff_val
    binding = "likely binder" if (aff_prob or 0) > 0.5 else "likely non-binder"
    if ic50 < 0.01:
        potency = "sub-nM potency (extremely potent)"
    elif ic50 < 0.1:
        potency = "nM range (very potent)"
    elif ic50 < 1.0:
        potency = "low nM range (potent)"
    elif ic50 < 10.0:
        potency = "µM range (moderate)"
    else:
        potency = "high µM range (weak binder)"
    return f"Predicted IC50 ≈ {ic50:.4f} µM — {potency}. Classification: {binding}."


# ─────────────────────────────────────────────────────────────────────────────
# Evo2-40B — DNA Foundation Model (Generation)
# ─────────────────────────────────────────────────────────────────────────────

@rate_limited("default")
@with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
async def generate_dna_evo2(
    sequence: str,
    num_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 4,
    top_p: float = 1.0,
    enable_logits: bool = False,
    num_generations: int = 1,
) -> dict[str, Any]:
    """
    Generate DNA sequences using Arc Evo2-40B via NVIDIA NIM.

    Evo2 is a 40-billion parameter biological foundation model trained on
    millions of genomic sequences. It models long-range dependencies across
    entire genomes while retaining single-nucleotide sensitivity — making it
    ideal for regulatory element design, gene synthesis, and sequence
    completion.

    Args:
        sequence:       Input DNA seed sequence (5'→3', ACGT only).
                        Evo2 continues from this sequence.
        num_tokens:     Number of new tokens to generate (1–1200). Default 200.
        temperature:    Sampling temperature (0.0–1.0).
                        Lower = more deterministic/conservative.
                        Higher = more diverse/creative. Default 1.0.
        top_k:          Top-K sampling (0–6). 0 = disabled, 4 = standard for DNA.
        top_p:          Top-P nucleus sampling (0.0–1.0). Default 1.0.
        enable_logits:  Return per-token logit scores (useful for sequence scoring).
        num_generations:Number of independent generation runs (1–5). Default 1.

    Returns:
        {
          status, model, seed_sequence_length,
          generations: [
            {
              generated_sequence,    # New DNA bases only
              full_sequence,         # Seed + generated
              length_generated,
              gc_content_pct,        # GC% of generated region
              logits                 # Optional: per-position logit scores
            }
          ],
          parameters_used,
          biological_context
        }

    Biological use cases:
        • Regulatory element design (promoters, enhancers)
        • Gene sequence completion / repair
        • Synthetic biology: novel codon-optimized sequences
        • Variant effect scoring (compare logits of WT vs mutant)
        • CRISPR guide RNA target region analysis
    """
    sequence    = BioValidator.validate_sequence(sequence, "nucleotide")
    num_tokens  = BioValidator.clamp_int(num_tokens, 1, 1200, "num_tokens")
    top_k       = BioValidator.clamp_int(top_k, 0, 6, "top_k")
    num_generations = BioValidator.clamp_int(num_generations, 1, 5, "num_generations")

    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"temperature must be 0.0–1.0, got {temperature}")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be 0.0–1.0, got {top_p}")

    client  = await get_http_client()
    headers = _evo2_headers()

    payload = {
        "sequence":                 sequence,
        "num_tokens":               num_tokens,
        "temperature":              temperature,
        "top_k":                    top_k,
        "top_p":                    top_p,
        "enable_logits_reporting":  enable_logits,
    }

    logger.info(
        f"[Evo2-40B] Generating {num_tokens} DNA tokens "
        f"(T={temperature}, top_k={top_k}, runs={num_generations})"
    )

    # Run multiple generations in parallel
    async def _single_generation() -> dict[str, Any]:
        t0   = time.monotonic()
        resp = await client.post(
            _EVO2_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        if resp.status_code == 402:
            raise PermissionError("NVIDIA NIM quota exceeded.")
        if resp.status_code == 401:
            raise PermissionError("Invalid NVIDIA API key.")
        resp.raise_for_status()
        data    = resp.json()
        elapsed = round(time.monotonic() - t0, 2)

        generated = data.get("sequence", "")
        full_seq  = sequence + generated

        return {
            "generated_sequence":  generated,
            "full_sequence":       full_seq,
            "length_generated":    len(generated),
            "gc_content_pct":      _gc_content(generated),
            "elapsed_s":           elapsed,
            "logits":              data.get("logits") if enable_logits else None,
        }

    generations_raw = await asyncio.gather(
        *[_single_generation() for _ in range(num_generations)],
        return_exceptions=True,
    )

    generations: list[dict] = []
    for i, g in enumerate(generations_raw):
        if isinstance(g, Exception):
            logger.warning(f"[Evo2] Generation {i + 1} failed: {g}")
            generations.append({"error": str(g), "run": i + 1})
        else:
            g["run"] = i + 1
            generations.append(g)

    return {
        "status":                "success",
        "model":                 "Arc Evo2-40B via NVIDIA NIM",
        "seed_sequence":         sequence[:60] + ("..." if len(sequence) > 60 else ""),
        "seed_sequence_length":  len(sequence),
        "generations":           generations,
        "parameters_used": {
            "num_tokens":   num_tokens,
            "temperature":  temperature,
            "top_k":        top_k,
            "top_p":        top_p,
            "logits":       enable_logits,
        },
        "biological_context": {
            "model_scale":       "40 billion parameters",
            "training_data":     "Millions of genomic sequences across domains of life",
            "key_capabilities": [
                "Long-range genomic context modeling",
                "Single-nucleotide mutation sensitivity",
                "Regulatory element design",
                "Gene sequence completion",
                "Variant effect prediction (via logits)",
            ],
            "gc_content_note": (
                "Typical mammalian coding regions: 50–60% GC. "
                "Promoters/CpG islands: 60–70% GC."
            ),
        },
    }


def _gc_content(seq: str) -> float:
    """Calculate GC% of a DNA sequence."""
    if not seq:
        return 0.0
    gc = seq.upper().count("G") + seq.upper().count("C")
    return round(gc / len(seq) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Evo2-40B — Sequence Scoring (variant effect prediction)
# ─────────────────────────────────────────────────────────────────────────────

@rate_limited("default")
@with_retry(max_attempts=3)
async def score_sequence_evo2(
    wildtype_sequence: str,
    variant_sequence: str,
) -> dict[str, Any]:
    """
    Score two DNA sequences with Evo2-40B and compare their log-likelihoods.

    This is the core operation for variant effect prediction — compare a
    wildtype sequence against a mutant to estimate functional impact.

    Args:
        wildtype_sequence: Reference/wildtype DNA sequence.
        variant_sequence:  Mutant/variant DNA sequence (same length).

    Returns:
        {
          wildtype_score,    # Mean log-likelihood under Evo2
          variant_score,     # Mean log-likelihood under Evo2
          delta_score,       # variant - wildtype (negative = likely deleterious)
          interpretation,    # Human-readable effect prediction
          mutation_positions # List of positions where sequences differ
        }
    """
    wt  = BioValidator.validate_sequence(wildtype_sequence, "nucleotide")
    var = BioValidator.validate_sequence(variant_sequence,  "nucleotide")

    if len(wt) != len(var):
        raise ValueError(
            f"Wildtype ({len(wt)} bp) and variant ({len(var)} bp) must be "
            "the same length for scoring comparison."
        )

    # Find mutation positions
    mutations = [i for i, (w, v) in enumerate(zip(wt, var)) if w != v]
    if not mutations:
        return {
            "status":             "no_mutations",
            "message":            "Wildtype and variant sequences are identical.",
            "mutation_positions": [],
        }

    client  = await get_http_client()
    headers = _evo2_headers()

    # Score both sequences — use logits to get per-token likelihoods
    async def _score(seq: str) -> float | None:
        resp = await client.post(
            _EVO2_URL,
            headers=headers,
            json={
                "sequence":                seq,
                "num_tokens":              1,     # Minimal generation
                "temperature":             1.0,
                "top_k":                   1,
                "enable_logits_reporting": True,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data   = resp.json()
        logits = data.get("logits")
        if logits and isinstance(logits, list):
            # Mean log-likelihood across positions
            return round(sum(logits) / len(logits), 6)
        return None

    wt_score, var_score = await asyncio.gather(_score(wt), _score(var))

    delta: float | None = None
    if wt_score is not None and var_score is not None:
        delta = round(var_score - wt_score, 6)

    return {
        "status":             "success",
        "model":              "Arc Evo2-40B via NVIDIA NIM",
        "wildtype_score":     wt_score,
        "variant_score":      var_score,
        "delta_score":        delta,
        "mutation_count":     len(mutations),
        "mutation_positions": mutations[:20],   # cap list
        "interpretation":     _interpret_variant_score(delta),
        "note": (
            "Negative delta = variant less likely under Evo2 model "
            "(potentially deleterious). "
            "Positive delta = variant more likely (potentially neutral/beneficial)."
        ),
    }


def _interpret_variant_score(delta: float | None) -> str:
    """Interpret delta log-likelihood score for a variant."""
    if delta is None:
        return "Could not compute delta score."
    if delta < -2.0:
        return "Strongly deleterious signal — large deviation from natural sequences."
    if delta < -0.5:
        return "Potentially deleterious — moderate deviation from wildtype."
    if delta < 0.0:
        return "Slightly deleterious — mild deviation from wildtype."
    if delta < 0.5:
        return "Likely neutral — within normal sequence variation."
    return "Potentially beneficial — higher likelihood than wildtype under Evo2."


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Drug Discovery Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def design_protein_ligand(
    uniprot_accession: str,
    ligand_smiles: str,
    predict_affinity: bool = True,
    method_conditioning: str | None = "x-ray",
) -> dict[str, Any]:
    """
    Full drug-discovery pipeline in one tool call.

    Steps (fully automated):
      1. Fetch protein sequence + metadata from UniProt
      2. Submit to Boltz-2 for structure + affinity prediction
      3. Return integrated report with structure, scores, and drug context

    Args:
        uniprot_accession: Target protein UniProt ID (e.g. 'P00533' for EGFR).
        ligand_smiles:     Drug/compound SMILES string.
        predict_affinity:  Compute binding affinity. Default True.
        method_conditioning: 'x-ray' | 'nmr' | 'md' | None. Default 'x-ray'.

    Returns:
        Integrated report: protein info + Boltz-2 structure + affinity scores.
    """
    from biomcp.tools.proteins import get_protein_info

    logger.info(
        f"[Drug Design] {uniprot_accession} + ligand: "
        f"{ligand_smiles[:30]}... affinity={predict_affinity}"
    )

    # Step 1 — Fetch protein
    protein_info = await get_protein_info(uniprot_accession)
    if "error" in protein_info:
        return {"error": f"UniProt lookup failed: {protein_info['error']}"}

    sequence = protein_info.get("sequence", "")
    if not sequence:
        return {"error": f"No sequence available for {uniprot_accession}."}

    # Step 2 — Boltz-2 prediction
    boltz_result = await predict_structure_boltz2(
        protein_sequences=[sequence],
        ligand_smiles=[ligand_smiles],
        predict_affinity=predict_affinity,
        method_conditioning=method_conditioning,
    )

    return {
        "pipeline":          "BioMCP Drug Discovery Pipeline",
        "target": {
            "uniprot_accession": uniprot_accession,
            "name":              protein_info.get("full_name", ""),
            "gene":              protein_info.get("gene_names", []),
            "organism":          protein_info.get("organism", ""),
            "sequence_length":   protein_info.get("sequence_length", 0),
            "diseases":          protein_info.get("diseases", [])[:3],
            "function":          (protein_info.get("function") or "")[:400],
        },
        "ligand": {
            "smiles":            ligand_smiles,
            "note":              "Validate ADMET properties with get_compound_info if ChEMBL ID available.",
        },
        "structure_prediction":  boltz_result,
        "next_steps": [
            "Visualize structure at https://molstar.org/viewer/",
            "Run ADMET prediction: use get_compound_info tool",
            "Check clinical trials: use search_clinical_trials tool",
            "Find similar approved drugs: use get_drug_targets tool",
        ],
    }
