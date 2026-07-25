"""Retrieve canonical protein sequences from the UniProt REST API.

Generic: works for any UniProt accession, not specific to APOE.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict

import requests

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class SequenceRecord:
    accession: str
    entry_name: str
    description: str
    sequence: str
    length: int
    retrieved_utc: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def fetch_sequence(accession: str, timeout: int = DEFAULT_TIMEOUT) -> SequenceRecord:
    """Fetch the canonical sequence for a UniProt accession."""
    url = UNIPROT_FASTA_URL.format(accession=accession)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    lines = response.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Unexpected FASTA payload for {accession}: {response.text[:200]!r}")

    header = lines[0][1:]
    sequence = "".join(line.strip() for line in lines[1:])

    header_fields = header.split("|")
    entry_name = header_fields[2].split(" ", 1)[0] if len(header_fields) >= 3 else accession
    description = header_fields[2].split(" ", 1)[1] if len(header_fields) >= 3 else header

    return SequenceRecord(
        accession=accession,
        entry_name=entry_name,
        description=description,
        sequence=sequence,
        length=len(sequence),
        retrieved_utc=_utc_now(),
        source_url=url,
    )


def fetch_signal_peptide(accession: str, timeout: int = DEFAULT_TIMEOUT) -> dict | None:
    """Return the annotated signal-peptide feature for an accession, if UniProt has one.

    Used to resolve pre-protein vs mature-protein residue numbering: mature position
    n corresponds to pre-protein position n + len(signal peptide).
    """
    url = UNIPROT_JSON_URL.format(accession=accession)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    for feature in payload.get("features", []):
        if feature.get("type") == "Signal":
            location = feature["location"]
            start = location["start"]["value"]
            end = location["end"]["value"]
            return {
                "start": start,
                "end": end,
                "length": end - start + 1,
                "source_url": url,
            }
    return None


def apply_substitutions(sequence: str, substitutions: dict[int, str]) -> str:
    """Return `sequence` with 1-based positions replaced by the given residues.

    Raises if a position is out of range or a residue is not a single character.
    """
    residues = list(sequence)
    for position, residue in substitutions.items():
        if not 1 <= position <= len(residues):
            raise IndexError(f"Position {position} outside sequence of length {len(residues)}")
        if len(residue) != 1:
            raise ValueError(f"Substitution at {position} must be a single residue, got {residue!r}")
        residues[position - 1] = residue
    return "".join(residues)
