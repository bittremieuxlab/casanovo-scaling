import sys
import os
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add scripts directory to path to import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

import evaluate
# Import ExperimentEvaluator directly
from evaluate import ExperimentEvaluator, DynamicMassDict
import unittest.mock

class TestEvaluate:
    @pytest.fixture
    def evaluator(self):
        # Patch _load_unimod so it returns None instead of trying to hit the network
        # Since AA_MASSES was removed from module level, we are safe to instantiate here
        with unittest.mock.patch('evaluate.ExperimentEvaluator._load_unimod', return_value=None):
            yield ExperimentEvaluator()

    @pytest.fixture
    def aa_masses(self, evaluator):
        return evaluator.aa_masses


    def test_dynamic_mass_dict(self, aa_masses):
        d = DynamicMassDict(aa_masses)
        
        # Test existing AA
        assert d['A'] == aa_masses['A']
        
        # Test C+mass
        # Note: DynamicMassDict logic for 'C[+57.021]' style which matches what ptms_to_delta_mass produces in regex substitution?
        # Actually evaluate.py's _transform_match_ptm produces "[+57.021]". 
        # Then the sequence becomes "X[+57.021]".
        # re.split("(?<=.)(?=[A-Z])") splits this into "X[+57.021]" if X is capital?
        # Yes.
        
        token = "C[+57.021]"
        # aa_masses has "C": 103...
        expected = aa_masses['C'] + 57.021
        assert d[token] == pytest.approx(expected)
        
        # Test N-term like "+42.011"
        # The key "+42.011" is in the dict with value 42.010565
        assert d["+42.011"] == pytest.approx(42.010565)

    def test_format_sequence_and_scores_basic(self, evaluator):
        sequence = "PEPTIDE"
        scores = "1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        
        seq_out, scores_out = evaluator.format_sequence_and_scores(sequence, scores)
        
        assert seq_out == sequence
        assert scores_out == scores

    def test_format_sequence_and_scores_replacements(self, evaluator):
        # Test C[Carbamidomethyl] replacement
        sequence = "C[Carbamidomethyl]PEPTIDE"
        scores = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" # one score for C[Carbamidomethyl]
        
        seq_out, scores_out = evaluator.format_sequence_and_scores(sequence, scores)
        
        assert "C[UNIMOD:4]" in seq_out
        assert "C[Carbamidomethyl]" not in seq_out
        assert scores_out == scores

    def test_format_sequence_and_scores_nterm(self, evaluator):
        # Test N-term modification
        sequence = "[UNIMOD:1]PEPTIDE"
        scores = "0.9,0.8,0.7,0.6,0.5,0.4,0.3" # 7 scores for 7 AAs, but mod is attached
        
        # The function _add_n_term_token_score duplicates the first score for the N-term mod
        seq_out, scores_out = evaluator.format_sequence_and_scores(sequence, scores)
        
        assert seq_out == sequence
        # The evaluator adds the score to the front
        assert scores_out.startswith("0.9,0.9")

    def test_calculate_metrics_perfect_match(self, evaluator):
        # Create mock output data
        # output_data needs: sequence, sequence_true, aa_scores, score
        data = {
            "sequence": ["PEPTIDE"],
            "sequence_true": ["PEPTIDE"],
            "aa_scores": ["1.0,1.0,1.0,1.0,1.0,1.0,1.0"],
            "score": [1.0] # search_engine_score
        }
        output_data = pd.DataFrame(data)
        
        metrics = evaluator.calculate_metrics(output_data)
        
        assert metrics["N sequences"] == 1
        assert metrics["N predicted"] == 1
        assert metrics["AA precision"] == pytest.approx(1.0)
        assert metrics["AA recall"] == pytest.approx(1.0)
        assert metrics["Pep precision"] == pytest.approx(1.0)

    def test_calculate_metrics_partial_match(self, evaluator):
        # sequence_true has 7 AAs
        # sequence has 7 AAs, but one is wrong (A instead of E) mass-wise
        # P: 97.05, E: 129.04, A: 71.03
        data = {
            "sequence": ["PAPTIDE"],
            "sequence_true": ["PEPTIDE"], 
            "aa_scores": ["1.0,1.0,1.0,1.0,1.0,1.0,1.0"],
            "score": [1.0]
        }
        output_data = pd.DataFrame(data)
        
        metrics = evaluator.calculate_metrics(output_data)
        
        # PAPTIDE vs PEPTIDE
        # P match
        # A vs E mismatch
        # P match
        # T match
        # I match
        # D match
        # E match
        # 6 matches out of 7
        
        assert metrics["AA precision"] == pytest.approx(6/7)
        assert metrics["AA recall"] == pytest.approx(6/7)
        assert metrics["Pep precision"] == 0.0

    def test_calculate_metrics_no_prediction(self, evaluator):
        data = {
            "sequence": [None],
            "sequence_true": ["PEPTIDE"],
            "aa_scores": [None],
            "score": [None]
        }
        output_data = pd.DataFrame(data)
        
        metrics = evaluator.calculate_metrics(output_data)
        
        assert metrics["N sequences"] == 1
        assert metrics["N predicted"] == 0
        assert metrics["AA recall"] == 0.0

    def test_format_sequence_and_scores_unknown_ptm(self, evaluator):
        # Unknown PTMs are those not in REPLACEMENTS
        # The function should leave them as is, but it might print a warning (which we won't capture here)
        sequence = "M[Unknown]PEPTIDE"
        scores = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        
        seq_out, scores_out = evaluator.format_sequence_and_scores(sequence, scores)
        
        assert seq_out == sequence
        assert scores_out == scores

    def test_calculate_metrics_empty_df(self, evaluator):
        output_data = pd.DataFrame(columns=["sequence", "sequence_true", "aa_scores", "score"])
        
        metrics = evaluator.calculate_metrics(output_data)
        
        assert metrics["N sequences"] == 0
        assert metrics["N predicted"] == 0
        assert metrics["AA precision"] == 0.0
        assert metrics["AA recall"] == 0.0
        assert metrics["Pep precision"] == 0.0
