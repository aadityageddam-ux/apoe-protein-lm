"""Predict protein structure with ESMFold via the free ESM Metagenomic Atlas API.

No local GPU or model weights required. Generic: takes any amino-acid sequence.

Reference: Lin Z, Akin H, Rao R, et al. Science. 2023;379(6637):1123-1130.
PMID: 36927031 (ESMFold / ESM Metagenomic Atlas).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

# The public ESMFold endpoint rejects sequences longer than this.
MAX_SEQUENCE_LENGTH = 400

DEFAULT_TIMEOUT = 600
MAX_RETRIES = 4


class FoldingAPIError(RuntimeError):
    pass


def fold_sequence(
    sequence: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Fold `sequence` with ESMFold and return the predicted structure as PDB text."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Empty sequence")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds the public ESMFold limit of "
            f"{MAX_SEQUENCE_LENGTH} residues"
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                ESMFOLD_URL,
                data=sequence,
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(5 * (attempt + 1))
            continue

        if response.status_code == 200:
            pdb_text = response.text
            if "ATOM" not in pdb_text:
                raise FoldingAPIError(f"Response contained no ATOM records: {pdb_text[:200]!r}")
            return pdb_text

        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        time.sleep(5 * (attempt + 1))

    raise FoldingAPIError(f"ESMFold API failed after {max_retries} attempts: {last_error}")


def plddt_values(pdb_text: str) -> np.ndarray:
    """Per-residue pLDDT confidence on a 0-100 scale, from the CA B-factor column.

    The public ESMFold endpoint emits pLDDT on a 0-1 scale rather than the 0-100
    scale used elsewhere, so values are rescaled when they clearly fall in [0, 1].
    """
    values = [
        float(line[60:66])
        for line in pdb_text.splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]
    if not values:
        raise FoldingAPIError("No CA atoms found in PDB text")
    array = np.asarray(values)
    return array * 100.0 if array.max() <= 1.0 else array


def mean_plddt(pdb_text: str) -> float:
    """Mean per-residue pLDDT confidence (0-100)."""
    return float(plddt_values(pdb_text).mean())


def save_pdb(pdb_text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_text, encoding="utf-8")
    return path


def ca_coordinates(pdb_text: str) -> np.ndarray:
    """(N, 3) array of alpha-carbon coordinates in residue order."""
    coords = [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in pdb_text.splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]
    if not coords:
        raise FoldingAPIError("No CA atoms found in PDB text")
    return np.asarray(coords)


def superpose(
    mobile_pdb: str,
    reference_pdb: str,
    subset: np.ndarray | None = None,
) -> tuple[str, float, np.ndarray]:
    """Least-squares superpose `mobile_pdb` onto `reference_pdb` over CA atoms.

    Assumes a 1:1 residue correspondence (same-length sequences, as with point
    variants of one protein). Returns the transformed PDB text, the CA RMSD, and
    the per-residue deviation after superposition.

    `subset` gives 0-based residue indices to fit on — useful for restricting the
    fit to confidently predicted regions. RMSD is reported over the same subset;
    the returned deviations always cover every residue.
    """
    mobile_ca = ca_coordinates(mobile_pdb)
    reference_ca = ca_coordinates(reference_pdb)
    if mobile_ca.shape != reference_ca.shape:
        raise ValueError(
            f"Residue count mismatch: {mobile_ca.shape[0]} vs {reference_ca.shape[0]}"
        )

    fit = slice(None) if subset is None else np.asarray(subset)
    mobile_centre = mobile_ca[fit].mean(axis=0)
    reference_centre = reference_ca[fit].mean(axis=0)

    # Kabsch algorithm: optimal rotation minimising RMSD between two point sets.
    covariance = (mobile_ca[fit] - mobile_centre).T @ (reference_ca[fit] - reference_centre)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T

    aligned_ca = (mobile_ca - mobile_centre) @ rotation.T + reference_centre
    deviations = np.linalg.norm(aligned_ca - reference_ca, axis=1)
    rmsd = float(np.sqrt((deviations[fit] ** 2).mean()))

    transformed_lines = []
    for line in mobile_pdb.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            new_xyz = (xyz - mobile_centre) @ rotation.T + reference_centre
            line = f"{line[:30]}{new_xyz[0]:8.3f}{new_xyz[1]:8.3f}{new_xyz[2]:8.3f}{line[54:]}"
        transformed_lines.append(line)

    return "\n".join(transformed_lines) + "\n", rmsd, deviations
