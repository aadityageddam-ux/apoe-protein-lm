"""Offline unit tests for the pipeline's pure functions.

No network, no API token, no model weights: these test the parts whose correct
answers are known in advance -- substitution, PDB parsing, pLDDT rescaling and
the Kabsch superposition -- against hand-constructed inputs.

The committed results are checked separately, by `scripts/verify_results.py`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_sequence import apply_substitutions
from src.fold_sequence import (
    FoldingAPIError,
    ca_coordinates,
    mean_plddt,
    plddt_values,
    superpose,
)
from src.score_variant import PositionScore, total_score


# PDB stores coordinates to three decimals, so a text round trip is only good
# to about 1e-3 A. Nothing here can be asserted tighter than that.
PDB_COORDINATE_TOLERANCE = 1e-2


def pdb_from_coordinates(coords: np.ndarray, bfactors: list[float] | None = None) -> str:
    """Minimal CA-only PDB text, in the fixed-column format the parsers expect.

    The PDB coordinate columns hold three decimal places, so any round trip
    through this text loses precision at about 1e-3 A. Tolerances below are set
    against that floor, not against float precision.
    """
    bfactors = bfactors if bfactors is not None else [50.0] * len(coords)
    lines = []
    for i, ((x, y, z), b) in enumerate(zip(coords, bfactors), start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C"
        )
    return "\n".join(lines) + "\n"


class TestApplySubstitutions:
    def test_substitutes_at_one_based_positions(self):
        assert apply_substitutions("ACDEF", {1: "W"}) == "WCDEF"
        assert apply_substitutions("ACDEF", {5: "W"}) == "ACDEW"

    def test_applies_several_substitutions(self):
        assert apply_substitutions("ACDEF", {1: "W", 3: "Y"}) == "WCYEF"

    def test_leaves_the_original_untouched(self):
        original = "ACDEF"
        apply_substitutions(original, {1: "W"})
        assert original == "ACDEF"

    def test_substituting_the_same_residue_is_a_no_op(self):
        assert apply_substitutions("ACDEF", {2: "C"}) == "ACDEF"

    @pytest.mark.parametrize("position", [0, 6, -1, 100])
    def test_rejects_out_of_range_positions(self, position):
        with pytest.raises(IndexError):
            apply_substitutions("ACDEF", {position: "W"})

    @pytest.mark.parametrize("residue", ["", "WW", "ALA"])
    def test_rejects_multi_character_residues(self, residue):
        with pytest.raises(ValueError):
            apply_substitutions("ACDEF", {1: residue})


class TestPlddtParsing:
    def test_reads_the_b_factor_column(self):
        pdb = pdb_from_coordinates(np.zeros((3, 3)), bfactors=[80.0, 60.0, 40.0])
        assert plddt_values(pdb).tolist() == [80.0, 60.0, 40.0]
        assert mean_plddt(pdb) == pytest.approx(60.0)

    def test_rescales_values_emitted_on_a_zero_to_one_scale(self):
        pdb = pdb_from_coordinates(np.zeros((3, 3)), bfactors=[0.8, 0.6, 0.4])
        assert plddt_values(pdb).tolist() == pytest.approx([80.0, 60.0, 40.0])

    def test_does_not_rescale_when_a_value_exceeds_one(self):
        pdb = pdb_from_coordinates(np.zeros((2, 3)), bfactors=[1.0, 95.0])
        assert plddt_values(pdb).tolist() == [1.0, 95.0]

    def test_raises_when_there_are_no_ca_atoms(self):
        with pytest.raises(FoldingAPIError):
            plddt_values("HEADER    NOT A STRUCTURE\nEND\n")


class TestCaCoordinates:
    def test_parses_coordinates_in_residue_order(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert ca_coordinates(pdb_from_coordinates(coords)) == pytest.approx(coords)

    def test_raises_when_there_are_no_ca_atoms(self):
        with pytest.raises(FoldingAPIError):
            ca_coordinates("END\n")


class TestSuperpose:
    @staticmethod
    def _reference(n: int = 12, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(n, 3)) * 10.0

    @staticmethod
    def _rotation(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def test_identical_structures_superpose_to_zero(self):
        pdb = pdb_from_coordinates(self._reference())
        _, rmsd, deviations = superpose(pdb, pdb)
        assert rmsd == pytest.approx(0.0, abs=1e-9)
        assert deviations.max() == pytest.approx(0.0, abs=1e-9)

    def test_recovers_a_pure_rotation_and_translation(self):
        # A rigid-body move changes no internal geometry, so RMSD must come back to zero.
        reference = self._reference()
        moved = reference @ self._rotation(0.7).T + np.array([25.0, -13.0, 4.0])
        _, rmsd, _ = superpose(
            pdb_from_coordinates(moved), pdb_from_coordinates(reference)
        )
        assert rmsd == pytest.approx(0.0, abs=PDB_COORDINATE_TOLERANCE)

    def test_does_not_use_a_reflection_to_cheat(self):
        # A mirrored structure is genuinely different; an unguarded SVD would
        # "fit" it with a determinant -1 rotation and report RMSD 0.
        reference = self._reference()
        mirrored = reference * np.array([1.0, 1.0, -1.0])
        _, rmsd, _ = superpose(
            pdb_from_coordinates(mirrored), pdb_from_coordinates(reference)
        )
        assert rmsd > 1.0

    def test_known_displacement_gives_the_expected_rmsd(self):
        reference = self._reference()
        shifted = reference.copy()
        shifted[0] += np.array([3.0, 0.0, 0.0])
        _, rmsd, deviations = superpose(
            pdb_from_coordinates(shifted), pdb_from_coordinates(reference)
        )
        # One residue moved by 3 A out of n; the fit absorbs part of it, so the
        # RMSD must be positive but below the raw single-residue displacement.
        assert 0.0 < rmsd < 3.0
        assert deviations.shape == (len(reference),)

    def test_subset_restricts_the_fit_and_the_rmsd(self):
        reference = self._reference()
        mobile = reference.copy()
        mobile[6:] += np.array([20.0, 0.0, 0.0])  # only the tail moves
        core = np.arange(6)
        _, global_rmsd, _ = superpose(
            pdb_from_coordinates(mobile), pdb_from_coordinates(reference)
        )
        _, core_rmsd, deviations = superpose(
            pdb_from_coordinates(mobile), pdb_from_coordinates(reference), subset=core
        )
        assert core_rmsd == pytest.approx(0.0, abs=PDB_COORDINATE_TOLERANCE)
        assert core_rmsd < global_rmsd
        # Deviations are always reported for every residue, not just the subset.
        assert deviations.shape == (len(reference),)
        assert deviations[6:].min() > 1.0

    def test_rejects_a_residue_count_mismatch(self):
        with pytest.raises(ValueError):
            superpose(
                pdb_from_coordinates(self._reference(n=10)),
                pdb_from_coordinates(self._reference(n=11)),
            )

    def test_transformed_output_is_still_parseable_pdb(self):
        reference = self._reference()
        moved = reference @ self._rotation(1.1).T + 5.0
        transformed, _, _ = superpose(
            pdb_from_coordinates(moved), pdb_from_coordinates(reference)
        )
        assert ca_coordinates(transformed) == pytest.approx(reference, abs=PDB_COORDINATE_TOLERANCE)


class TestTotalScore:
    def test_sums_log_probabilities(self):
        scores = [
            PositionScore(position=130, residue="C", log_prob=-7.72, prob=math.exp(-7.72)),
            PositionScore(position=176, residue="C", log_prob=-8.70, prob=math.exp(-8.70)),
        ]
        assert total_score(scores) == pytest.approx(-16.42)

    def test_empty_score_list_totals_zero(self):
        assert total_score([]) == 0.0

    def test_log_prob_and_prob_stay_consistent(self):
        score = PositionScore(position=1, residue="A", log_prob=math.log(0.25), prob=0.25)
        assert math.exp(score.log_prob) == pytest.approx(score.prob)
