"""Run this before believing any number in this repository.

It re-derives, from the committed artifacts alone (`results/`, `data/variants.json`):

* the three variant sequences differ from each other at exactly the two claimed
  positions, and carry the residues the allele definitions require;
* the numbering convention recorded in `variants.json` agrees with the
  signal-peptide offset and with the dbSNP reference residues;
* every score in `scores.csv` is the sum of its own per-residue log-probabilities;
* the model ranking and the clinical ranking are the ones the README states,
  and they are exactly inverted;
* the Arg-over-Cys probability ratios quoted in the README follow from the
  committed log-probabilities;
* the pLDDT means, the fraction of confident residues, the size of the
  confidently predicted core and the Ca RMSDs all match the committed PDBs;
* every number quoted in README.md matches the artifact it came from.

It re-derives rather than re-reads on purpose. A check that trusted the same
summary the README was written from would prove nothing.

This verifies internal consistency of the committed run. It does not re-run the
Hugging Face or ESMFold API calls, so it cannot tell you that a fresh run today
would produce the same numbers -- see the reproducibility limitation in README.md.

Exit 0 means every check held. Anything else means it did not.

    python scripts/verify_results.py
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fold_sequence import ca_coordinates, plddt_values, superpose  # noqa: E402

ALLELES = ("e2", "e3", "e4")

# The allele definitions, from the two e-defining SNPs. Pre-protein numbering.
EXPECTED_RESIDUES = {
    "e2": {130: "C", 176: "C"},
    "e3": {130: "C", 176: "R"},
    "e4": {130: "R", 176: "R"},
}

# Clinical direction, from the human cohort evidence cited in the README.
# Rank 1 = most favourable. Not a model output; the comparator the model is scored against.
CLINICAL_RANK = {"e2": 1, "e3": 2, "e4": 3}

PLDDT_THRESHOLD = 70.0

failures: list[str] = []
checks = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok    {label}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))
        failures.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    variants = json.loads((ROOT / "data" / "variants.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((ROOT / "results" / "scores.csv").open(encoding="utf-8")))
    by_allele = {r["allele"]: r for r in rows}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # ------------------------------------------------------------------ sequences
    section("Sequences and numbering")

    # variants.json keys the three sequences as "APOE-e2" / "APOE-e3" / "APOE-e4".
    sequences = {
        name.split("-")[-1]: record["sequence"] for name, record in variants["variants"].items()
    }
    check(set(sequences) == set(ALLELES), "three alleles present", ", ".join(sorted(sequences)))

    length = variants["sequence_length"]
    check(
        all(len(s) == length for s in sequences.values()),
        "all sequences are the recorded length",
        f"{length} residues",
    )

    signal = variants["signal_peptide"]
    check(
        signal["length"] == signal["end"] - signal["start"] + 1,
        "signal-peptide length is self-consistent",
        f"{signal['start']}-{signal['end']} = {signal['length']}",
    )

    # Mature numbering (112/158) + signal peptide length == pre-protein numbering (130/176).
    check(
        112 + signal["length"] == 130 and 158 + signal["length"] == 176,
        "mature->pre-protein offset reconciles both conventions",
        f"112+{signal['length']}=130, 158+{signal['length']}=176",
    )

    dbsnp = variants["dbsnp_verification"]
    reference = sequences["e3"]
    for rsid, record in dbsnp.items():
        position = record["position"]
        check(
            reference[position - 1] == record["reference_residue"],
            f"{rsid}: reference residue matches the downloaded sequence",
            f"pos {position} = {reference[position - 1]}",
        )

    # The variants must differ at the two SNP positions and nowhere else.
    snp_positions = sorted(record["position"] for record in dbsnp.values())
    for a, b in itertools.combinations(ALLELES, 2):
        differing = [
            i + 1 for i, (x, y) in enumerate(zip(sequences[a], sequences[b])) if x != y
        ]
        check(
            set(differing) <= set(snp_positions),
            f"{a} vs {b} differ only at the SNP positions",
            f"differing: {differing or 'none'}",
        )

    for allele, expected in EXPECTED_RESIDUES.items():
        actual = {p: sequences[allele][p - 1] for p in expected}
        check(actual == expected, f"{allele} carries its defining residues", str(actual))

    # --------------------------------------------------------------------- scores
    section("Scores")

    for allele in ALLELES:
        row = by_allele[allele]
        total = float(row["log_prob_130"]) + float(row["log_prob_176"])
        check(
            close(total, float(row["masked_marginal_total"]), 1e-9),
            f"{allele}: masked-marginal total is the sum of its parts",
            f"{total:.4f}",
        )
        check(
            row["residue_130"] == EXPECTED_RESIDUES[allele][130]
            and row["residue_176"] == EXPECTED_RESIDUES[allele][176],
            f"{allele}: scored residues match the allele definition",
            f"{row['residue_130']}130 / {row['residue_176']}176",
        )
        check(
            int(row["clinical_rank"]) == CLINICAL_RANK[allele],
            f"{allele}: clinical rank is the cited one",
            row["clinical_rank"],
        )

    model_order = sorted(ALLELES, key=lambda a: -float(by_allele[a]["masked_marginal_total"]))
    check(
        list(model_order) == ["e4", "e3", "e2"],
        "model ranking is e4 > e3 > e2",
        " > ".join(model_order),
    )
    clinical_order = sorted(ALLELES, key=lambda a: CLINICAL_RANK[a])
    check(
        list(model_order) == list(reversed(clinical_order)),
        "model ranking is exactly inverted relative to the clinical ranking",
        f"model {' > '.join(model_order)} vs clinical {' > '.join(clinical_order)}",
    )
    for allele in ALLELES:
        derived = sorted(
            ALLELES, key=lambda a: -float(by_allele[a]["masked_marginal_total"])
        ).index(allele) + 1
        check(
            int(by_allele[allele]["model_rank"]) == derived,
            f"{allele}: recorded model rank matches the scores",
            str(derived),
        )

    # The README explains the inversion by the Arg:Cys probability ratio at each site.
    ratio_130 = math.exp(
        float(by_allele["e4"]["log_prob_130"]) - float(by_allele["e2"]["log_prob_130"])
    )
    ratio_176 = math.exp(
        float(by_allele["e4"]["log_prob_176"]) - float(by_allele["e2"]["log_prob_176"])
    )
    quoted_130, quoted_176 = (
        float(x.replace(",", ""))
        for x in re.search(
            r"about ([\d,]+)x \(position 130\) and ([\d,]+)x \(position 176\)",
            readme.replace("×", "x"),
        ).groups()
    )
    check(
        close(ratio_130, quoted_130, 0.05 * quoted_130),
        "position 130 Arg:Cys ratio matches the README",
        f"derived {ratio_130:,.0f}x vs quoted {quoted_130:,.0f}x",
    )
    check(
        close(ratio_176, quoted_176, 0.05 * quoted_176),
        "position 176 Arg:Cys ratio matches the README",
        f"derived {ratio_176:,.0f}x vs quoted {quoted_176:,.0f}x",
    )

    # ----------------------------------------------------------------- structures
    section("Structures")

    pdbs = {a: (ROOT / "results" / f"APOE-{a}.pdb").read_text(encoding="utf-8") for a in ALLELES}
    plddt = {a: plddt_values(t) for a, t in pdbs.items()}

    for allele in ALLELES:
        check(
            len(plddt[allele]) == length,
            f"{allele}: one pLDDT value per residue",
            f"{len(plddt[allele])}",
        )
        check(
            ca_coordinates(pdbs[allele]).shape == (length, 3),
            f"{allele}: one CA coordinate per residue",
        )

    means = {a: float(plddt[a].mean()) for a in ALLELES}
    check(
        all(60.0 <= m <= 68.0 for m in means.values()),
        "mean pLDDT is the stated ~64 for every variant",
        ", ".join(f"{a}={means[a]:.1f}" for a in ALLELES),
    )

    confident_fraction = {a: 100.0 * float((plddt[a] > PLDDT_THRESHOLD).mean()) for a in ALLELES}
    check(
        all(15.0 <= f <= 21.0 for f in confident_fraction.values()),
        "fraction of residues above pLDDT 70 is the stated ~16-20%",
        ", ".join(f"{a}={confident_fraction[a]:.1f}%" for a in ALLELES),
    )

    core = np.where(
        (plddt["e2"] > PLDDT_THRESHOLD)
        & (plddt["e3"] > PLDDT_THRESHOLD)
        & (plddt["e4"] > PLDDT_THRESHOLD)
    )[0]
    quoted_core = int(re.search(r"over the (\d+) residues all three models", readme).group(1))
    check(
        len(core) == quoted_core,
        "size of the confidently predicted core matches the README",
        f"{len(core)} residues",
    )

    global_rmsd = {}
    core_rmsd = {}
    for a, b in itertools.combinations(ALLELES, 2):
        _, g, _ = superpose(pdbs[a], pdbs[b])
        _, c, _ = superpose(pdbs[a], pdbs[b], subset=core)
        global_rmsd[(a, b)] = g
        core_rmsd[(a, b)] = c

    low, high = (
        float(x)
        for x in re.search(
            r"differ by (\d+)-(\d+) (?:Å|A) global", readme.replace("–", "-")
        ).groups()
    )
    check(
        all(low <= v <= high for v in global_rmsd.values()),
        "global CA RMSDs fall in the range the README quotes",
        ", ".join(f"{a}/{b}={v:.2f}" for (a, b), v in global_rmsd.items()) + f" vs {low:.0f}-{high:.0f} A",
    )
    check(
        all(v < 1.5 for v in core_rmsd.values()),
        "core RMSDs collapse to the stated ~1 A",
        ", ".join(f"{a}/{b}={v:.2f}" for (a, b), v in core_rmsd.items()),
    )
    check(
        all(core_rmsd[k] < global_rmsd[k] for k in global_rmsd),
        "the difference is global, not local: core RMSD < global RMSD for every pair",
    )

    # --------------------------------------------------------- README score table
    section("README consistency")

    table = re.findall(
        r"\|\s*\*\*(?:ε|e)([234])\*\*\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*\*\*(\S+)\*\*\s*\|\s*(\S+)\s*\|",
        readme,
    )
    check(len(table) == 3, "README result table has three allele rows", f"{len(table)} found")

    for digit, lp130, lp176, total, delta in table:
        allele = f"e{digit}"
        row = by_allele[allele]
        for quoted, column, label in (
            (lp130, "log_prob_130", "log P(130)"),
            (lp176, "log_prob_176", "log P(176)"),
            (total, "masked_marginal_total", "total"),
            (delta, "delta_llr_vs_e3", "dLLR"),
        ):
            value = float(quoted.replace("−", "-").replace("+", ""))
            check(
                close(value, float(row[column]), 0.005),
                f"README {allele} {label} matches scores.csv",
                f"{value} vs {float(row[column]):.4f}",
            )

    # ------------------------------------------------------------------- verdict
    print(f"\n{'-' * 68}")
    if failures:
        print(f"FAILED: {len(failures)} of {checks} checks did not hold\n")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"All {checks} checks held.")
    print("Internal consistency only; the live API calls were not re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
