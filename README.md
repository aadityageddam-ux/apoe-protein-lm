# APOE variant scoring with a protein language model

**Does a protein language model that has never seen human aging or clinical data independently recover the best-replicated human longevity association?**

**Result: no.** ESM-2's zero-shot ranking of the three human APOE alleles is **ε4 > ε3 > ε2** — monotonically *inverted* relative to the established clinical direction, under two independent scoring formulations, by roughly three orders of magnitude in per-residue probability.

This is a negative result and is reported as such. It is a confirmatory computational experiment, not exploratory discovery — one protein, three known variants, one testable question.

---

## The question

APOE is the most replicated longevity-associated gene in humans ([PMID 26930295](https://pubmed.ncbi.nlm.nih.gov/26930295/)). Its three common alleles have decades of human evidence attached:

| Allele | Position 130 / 176 | Established association |
|---|---|---|
| **ε2** | Cys / Cys | Increased odds of extreme longevity; enriched in centenarians ([PMID 30060062](https://pubmed.ncbi.nlm.nih.gov/30060062/), [PMID 26930295](https://pubmed.ncbi.nlm.nih.gov/26930295/)) |
| **ε3** | Cys / Arg | Reference / most common allele |
| **ε4** | Arg / Arg | Major Alzheimer's risk factor; decreased odds of extreme longevity, increased mortality ([PMID 9343467](https://pubmed.ncbi.nlm.nih.gov/9343467/), [PMID 30060062](https://pubmed.ncbi.nlm.nih.gov/30060062/)) |

ESM-2 is trained on evolutionary protein sequence data. It has never seen aging data, clinical outcomes, or Alzheimer's labels. If its zero-shot "fitness" scores ranked these three variants ε2 > ε3 > ε4, that would be a striking independent recovery of a human clinical signal from sequence alone.

They do not.

## The result

| Allele | log P(res 130) | log P(res 176) | Masked-marginal total | ΔLLR vs ε3 | Model rank | Clinical rank |
|---|---|---|---|---|---|---|
| **ε2** | −7.72 | −8.70 | **−16.42** | −8.56 | 3 (worst) | 1 (best) |
| **ε3** | −7.72 | −0.14 | **−7.86** | 0.00 | 2 | 2 |
| **ε4** | −0.51 | −0.13 | **−0.64** | +7.21 | 1 (best) | 3 (worst) |

Model ranking: **ε4 > ε3 > ε2.** Clinical ranking: **ε2 > ε3 > ε4.** Exactly reversed.

The cause is visible directly in the model's predictions. With either site masked, ESM-2 assigns Arg — the ε4 residue at both positions — about 1,350× (position 130) and 5,200× (position 176) more probability than Cys. ε4 is Arg/Arg and therefore scores highest; ε2 is Cys/Cys and scores lowest.

The ranking rests entirely on these sequence scores. The structure predictions contribute nothing to it. ESMFold's confidence on APOE is low — mean pLDDT ≈ 64, with only ~16–20% of residues above the conventional 70 threshold — and while the variants differ by 6–10 Å global Cα RMSD, that collapses to ~1 Å over the 47 residues all three models predict confidently. The deviation is concentrated at the termini, in the pattern expected from the two domains being placed differently relative to one another rather than from any local change at residues 130 or 176. A single point substitution is not separable from single-sequence prediction noise at this confidence. **No structural claim is made in either direction**; the structures are included for completeness and visual inspection only.

### Interpretation

ESM-2's training objective is to predict residues from evolutionary sequence context, so its scores measure how *typical* a residue is in the protein universe it was trained on. That is not the same quantity as an effect on human healthspan, and here the two point in opposite directions.

The practical takeaway: **zero-shot protein-language-model fitness scores should not be treated as a proxy for human longevity effects at this locus.** A single case study cannot show this generalises, but it is a clean counterexample to the assumption.

One hypothesis for the inversion — that Arg is the residue favoured across the evolutionary distribution the model was trained on, regardless of its consequences in humans — is **explicitly flagged as an untested hypothesis**, not a claim. The citation set backing this project does not cover APOE allele ancestry or cross-species conservation, and testing it would require a separate analysis of APOE orthologues.

## Method

1. **Fetch** the canonical sequence from UniProt [P02649](https://www.uniprot.org/uniprotkb/P02649) at run time. Nothing is hardcoded.
2. **Resolve the numbering convention** (hard gate, see below).
3. **Construct** the ε2/ε3/ε4 sequences by substitution at the two verified positions.
4. **Score** each variant with ESM-2 (650M) zero-shot masked marginals via the free Hugging Face Inference API — mask a position, read off the log-probability of the residue actually present, sum over both positions. Cross-checked against the standard wild-type-context ΔLLR formulation.
5. **Fold** all three sequences with ESMFold via the free ESM Metagenomic Atlas API, and record pLDDT confidence.
6. **Superpose** the structures (Kabsch, over Cα atoms), globally and over the confidently predicted core, and visualise the overlay with py3Dmol.
7. **Report** the verdict, computed programmatically from the ranking rather than asserted.

### The numbering gate

APOE variant positions appear in the literature under two conventions — **mature-protein** numbering (112 / 158, after the 18-residue signal peptide is cleaved) and **pre-protein** numbering (130 / 176, signal peptide included). Getting this wrong would silently score the wrong residues and invalidate everything downstream, so the notebook resolves it before any scoring runs, using two independent live checks:

- **UniProt's signal-peptide annotation** confirms the 18-residue offset and that the 317-residue canonical sequence includes the signal peptide — so P02649 is written in pre-protein numbering.
- **dbSNP** records for the two ε-defining SNPs, queried live from NCBI E-utilities: `rs429358` = `NP_000032.1:p.Cys130Arg` and `rs7412` = `NP_000032.1:p.Arg176Cys`.

The notebook asserts that the residues dbSNP names as reference (Cys130, Arg176) are exactly what the downloaded sequence contains, and that the two positions match the signal-peptide offset. It halts if either check fails. **Resolved convention: pre-protein numbering, positions 130 and 176.** The canonical UniProt sequence is the ε3 allele.

## Reproducing it

Everything runs on free-tier resources: no paid API, no paid compute, no local GPU, no local model weights. The only credential is a free Hugging Face token (read scope is sufficient).

```bash
git clone https://github.com/aadityageddam-ux/apoe-protein-lm.git
```

```bash
cd apoe-protein-lm && pip install -r requirements.txt
```

Get a free token at <https://huggingface.co/settings/tokens>, then:

```bash
cp .env.example .env
```

Put your token in `.env` as `HF_TOKEN=...`, then run the notebook:

```bash
jupyter notebook notebook.ipynb
```

Run all cells. The pipeline makes 8 Hugging Face Inference API calls and 3 ESMFold calls; expect a few minutes end-to-end. No manual intervention is needed beyond the token.

To use the pipeline on a different protein, the `src/` functions are generic — they take any UniProt accession, any sequence, and any set of 1-based positions:

```python
from src.fetch_sequence import fetch_sequence, apply_substitutions
from src.score_variant import score_positions, total_score
from src.fold_sequence import fold_sequence, mean_plddt

record = fetch_sequence("P01308")                      # any accession
variant = apply_substitutions(record.sequence, {25: "A"})
print(total_score(score_positions(variant, [25])))     # any positions
print(mean_plddt(fold_sequence(variant)))
```

## Repository layout

```
apoe-protein-lm/
├── notebook.ipynb              # Full walkthrough: code, charts, structures, conclusion
├── src/
│   ├── fetch_sequence.py       # UniProt retrieval + substitution helpers
│   ├── score_variant.py        # ESM-2 zero-shot scoring (generic)
│   └── fold_sequence.py        # ESMFold API + PDB handling + superposition (generic)
├── data/
│   └── variants.json           # The 3 sequences + numbering provenance and dbSNP verification
├── results/
│   ├── APOE-e{2,3,4}.pdb       # Predicted structures
│   ├── scores.csv              # Fitness scores per variant
│   ├── scores.png              # Bar chart vs. clinical direction
│   ├── structure_deviation.png # Per-residue backbone deviation
│   └── structure_overlay.html  # Interactive py3Dmol overlay
├── README.md
└── requirements.txt
```

Note: GitHub's notebook renderer strips JavaScript, so the interactive 3D overlay will not display in-browser on GitHub. Open `results/structure_overlay.html` locally, or run the notebook, to view it. The static figures render fine.

## Limitations

- **The model has no aging or clinical signal.** ESM-2 and ESMFold were trained on general protein sequence and structure data, not on aging, mortality, or clinical-outcome data ([PMID 36927031](https://pubmed.ncbi.nlm.nih.gov/36927031/)). Any correspondence with clinical direction would have been a property of general sequence "naturalness", not a learned aging signal — and here there is none.
- **Fitness proxy ≠ disease risk.** Zero-shot mutation scores approximate evolutionary fitness / sequence plausibility, not Alzheimer's risk or lifespan. A match would have been suggestive, not confirmatory of mechanism; the mismatch likewise does not overturn the clinical evidence, which rests on large human cohorts ([PMID 30060062](https://pubmed.ncbi.nlm.nih.gov/30060062/) — 28,297 participants across 7 cohorts).
- **Single-sequence structure prediction is less certain — and here it was not confident enough to use.** ESMFold predicts from one sequence without a multiple sequence alignment — faster, but carrying more uncertainty than MSA-based methods such as AlphaFold2 ([PMID 36927031](https://pubmed.ncbi.nlm.nih.gov/36927031/)). On APOE it returned a mean pLDDT of roughly 64, with only a small minority of residues above the conventional confidence threshold. The predicted structures are reported for completeness and visual inspection only; they support no conclusion about whether the ε alleles differ in fold, and none is drawn.
- **n = 3.** One protein, three variants, one model. A single case study — not a benchmark, not a statistically powered test, and not a general claim about protein language models.
- **In silico only.** Nothing here is experimentally validated. Every number in this repository is a model prediction and constitutes no biological finding beyond "this is what the model predicts."
- **APOE biology is not captured by point-substitution scoring.** These scores treat the protein as an isolated sequence; the ε2 allele's effects have been characterised at the level of lipid metabolism in carriers ([PMID 35997888](https://pubmed.ncbi.nlm.nih.gov/35997888/)), a level of biology entirely outside what a single-sequence model represents.
- **Reproducibility is for the method, not bit-exact numbers.** Scores come from the Hugging Face Inference API, where I did not pin the model revision, so a rerun months later may differ slightly or fail if the hosted model changes. The committed `results/` and `data/variants.json` are the run behind every number above; I rechecked that the quoted ratios and totals follow from `results/scores.csv`, but I have not re-run the live API calls since the original run.

## Citations

1. Farrer LA, Cupples LA, Haines JL, et al. Effects of age, sex, and ethnicity on the association between apolipoprotein E genotype and Alzheimer disease: a meta-analysis. *JAMA.* 1997;278(16):1349-1356. PMID: [9343467](https://pubmed.ncbi.nlm.nih.gov/9343467/).
2. Sebastiani P, Gurinovich A, Nygaard M, et al. APOE Alleles and Extreme Human Longevity. *J Gerontol A Biol Sci Med Sci.* 2019;74(1):44-51. doi:[10.1093/gerona/gly174](https://doi.org/10.1093/gerona/gly174). PMID: [30060062](https://pubmed.ncbi.nlm.nih.gov/30060062/).
3. Ryu S, Atzmon G, Barzilai N, Raghavachari N, Suh Y. Genetic landscape of APOE in human longevity revealed by high-throughput sequencing. *Mech Ageing Dev.* 2016;155:7-9. doi:[10.1016/j.mad.2016.02.010](https://doi.org/10.1016/j.mad.2016.02.010). PMID: [26930295](https://pubmed.ncbi.nlm.nih.gov/26930295/).
4. Sebastiani P, Song Z, Ellis D, et al. A metabolomic signature of the APOE2 allele. *GeroScience.* 2022;45(1):415-426. doi:[10.1007/s11357-022-00646-9](https://doi.org/10.1007/s11357-022-00646-9). PMID: [35997888](https://pubmed.ncbi.nlm.nih.gov/35997888/).
5. Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science.* 2023;379(6637):1123-1130. doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574). PMID: [36927031](https://pubmed.ncbi.nlm.nih.gov/36927031/).

*All citations retrieved from PubMed on 2026-07-25.*

**Data sources.** UniProt [P02649](https://www.uniprot.org/uniprotkb/P02649) (sequence); NCBI dbSNP [rs429358](https://www.ncbi.nlm.nih.gov/snp/rs429358) and [rs7412](https://www.ncbi.nlm.nih.gov/snp/rs7412) (numbering verification). Both retrieved programmatically at run time — see `data/variants.json` for retrieval timestamps.

## License

Code and results: MIT (see `LICENSE`). Input data are retrieved from UniProt (CC BY 4.0) and NCBI dbSNP at run time; ESM-2 and ESMFold are by Meta AI (MIT-licensed releases) and are accessed through hosted free-tier APIs.
