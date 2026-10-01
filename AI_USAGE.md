# AI Usage

**Models:** Claude Opus 5, Claude Sonnet 5 and Claude Sonnet 5.5, via Claude Code. Every commit on `main` carries a Claude co-author trailer.

I used AI on this project from the first commit to the last, and I'd rather say exactly how than be vague. This file covers what I asked for, what the model did, what went wrong, and what was mine.

## What I asked for

The first session (2026-07-25) built from a PRD that I drafted with AI help and then edited. It fixed one question: does ESM-2, which has never seen aging or clinical data, recover the APOE longevity ranking? It limited the scope to one protein and three known alleles, and it set rules the build had to follow:

- **An honest-result requirement.** Report confirmation, partial confirmation or no confirmation exactly as the model showed it. No cherry-picking, and no reframing a negative result as a positive one.
- **A numbering gate.** APOE positions are 112/158 in mature-protein numbering and 130/176 when the signal peptide is counted, so the numbering had to be verified before any scoring code was written.
- **Citation traceability.** Every biological claim had to trace to a five-paper citation list, and anything else was out of scope.
- **A required limitations section** and a definition of done with no mocked or placeholder data.

I also told the session to ask me questions whenever it had a concern.

After the first build, my messages were about the figures: they didn't display, a notebook cell errored, the charts were cluttered, a legend covered the labels, and a line had no legend entry. I also asked for a plain-language summary at the bottom of the notebook.

In October 2026 I had the repository prepared for public release: a verification script, unit tests, CI, a license, and README corrections.

## What the model did

- Fetched the UniProt sequence and resolved the numbering against dbSNP (`p.Cys130Arg` and `p.Arg176Cys`, which is pre-protein numbering and matches UniProt), then built the three variants, scored them with ESM-2 by two formulations (masked marginal and a ΔLLR check), and predicted structures with ESMFold.
- Wrote the code in `src/`, the notebook and the charts.
- In October: `scripts/verify_results.py` (53 checks that re-derive every number in the README from the committed results), 27 offline unit tests, and CI on Python 3.11 and 3.13.
- Checked the ancestral-allele explanation of the result against PubMed and added it with its citation.

## What went wrong

**1. A conclusion written before the output was read.** The first draft of the notebook's conclusion said the predicted structures were "near-identical (sub-ångström backbone RMSD)". The executed output showed the opposite. The model caught it itself, in its own words "the Cα RMSDs are 6–10 Å, not the sub-ångström my draft conclusion claimed," and rewrote the claim. It also flagged that the pLDDT confidence was low, not high. The corrected text reports a 6–10 Å global deviation that collapses to about 1 Å over the 47 confidently predicted residues, and makes no structural claim in either direction.

**2. The corrected number was still slightly wrong.** In October the verification script re-derived the README's numbers from the committed files and found the global RMSD range was 6.10–10.89 Å, not "6–10 Å". The README now says 6–11 Å.

**3. Charts that didn't communicate.** The static figures didn't display, a notebook layout call errored, a legend sat over the "e2 vs e3" and "e4 vs e3" labels, and the pLDDT line had no legend entry. These were fixed over three commits (`a7851c5`, `782a94d`, `e9ad01d`).

**4. An explanation left as an "untested hypothesis".** The README said ε4 might score highest because Arg is favored in evolution, and said no citation covered it. The citation exists: ε4 is the ancestral allele (Hanlon & Rubinsztein 1995). Adding it turned the result from "unexplained inversion" into an expected one, and the phrase "clean counterexample" was toned down because n = 3 cannot support it.

**5. Small hygiene problems.** A local filesystem path was saved inside a notebook output (removed), the clone instructions had a placeholder (replaced), and the hosted model revision was never recorded, so the original run can't be exactly replayed. The README states that limit.

## What was mine and what wasn't

Mine: the question, the scope, the honest-result rule, the numbering gate and the citation rule, the decision to report the result as a negative one with n = 3, the requests that the figures be fixed, and the decision to publish.

Not mine: the code. Claude wrote the modules in `src/`, the notebook and the tests.

What the verification script does and doesn't prove: it shows that every number in the README matches the saved results, and that the saved results are internally consistent. It does not re-run the live ESM-2 and ESMFold calls.
