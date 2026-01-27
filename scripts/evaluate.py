import os
import pickle
import re
from typing import List, Dict, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from pyteomics import mztab

# Import evaluation functions from casanovo library
from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics

class DynamicMassDict:
    """
    A dictionary-like object that parses mass tokens on the fly.
    This avoids generating a massive dictionary of all possible mass shifts.
    """
    def __init__(self, aa_masses: Dict[str, float]):
        self.aa_masses = aa_masses

    def get(self, token: str, default: float = 0.0) -> float:
        try:
            return self[token]
        except (KeyError, ValueError):
            return default

    def __getitem__(self, token: str) -> float:
        if token in self.aa_masses:
            return self.aa_masses[token]
        
        # Try to parse if it's a complex token like 'C+57.021' or '+15.995'
        # The token format is typically AA+PTM or just PTM
        if len(token) == 1:
            return self.aa_masses[token] # Should have been caught above if in dict
        
        # Helper to safely float conversion
        def safe_float(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        # Handle N-term or other simple modifications if they are just numbers like '+42.011' or '-17.027'
        try:
             return float(token)
        except ValueError:
            pass

        # Parse AA+PTM format (e.g. M+15.995, C+57.021, C[+57.021])
        if '[' in token and token.endswith(']'):
            aa = token[0]
            ptm = token[token.find('[')+1:-1]
            return self.aa_masses.get(aa, 0.0) + safe_float(ptm)
            
        return self.aa_masses.get(token, 0.0)


class ExperimentEvaluator:
    def __init__(self):
        self.unimod_db = self._load_unimod()
        self.ptm_masses = {}
        self.aa_masses = self._init_aa_masses()
        self.dynamic_mass_dict = DynamicMassDict(self.aa_masses)
        self.replacements = self._init_replacements()
        
        # Compile regex patterns
        self.pep_split_pattern = re.compile(r"(?<=.)(?=[A-Z])")
        self.n_term_mod_pattern = re.compile(r"^((\[UNIMOD:[0-9]+\])+)")
        self.ptm_pattern = re.compile(r"\[UNIMOD:([0-9]+)\]")
        self.n_term_pattern_proforma = re.compile(r"^\[([0-9+-.]+)\]-([A-Z])(?:\[([0-9+-.]+)\])?")

    def _load_unimod(self):
        try:
            from pyteomics.mass.unimod import Unimod
            return Unimod()
        except ImportError:
            try:
                from pyteomics.mass import Unimod
                return Unimod()
            except ImportError:
                print("Warning: Could not import Unimod from pyteomics.mass")
                return None

    def _init_aa_masses(self) -> Dict[str, float]:
        # Copied from original code
        return {
            "G": 57.021463735,
            "A": 71.037113805,
            "S": 87.032028435,
            "P": 97.052763875,
            "V": 99.068413945,
            "T": 101.047678505,
            "C": 103.009184505,
            "C+57.021": 160.030644505,
            "L": 113.084064015,
            "I": 113.084064015,
            "N": 114.04292747,
            "D": 115.026943065,
            "Q": 128.05857754,
            "K": 128.09496305,
            "E": 129.042593135,
            "M": 131.040484645,
            "H": 137.058911875,
            "F": 147.068413945,
            "R": 156.10111105,
            "Y": 163.063328575,
            "W": 186.07931298,
            "+42.011": 42.010565,
            "+43.006": 43.005814,
            "-17.027": -17.026549,
            "+43.006-17.027": 25.980265,
            "M+15.995": 147.03539964499998,
            "N+0.984": 115.02694346999999,
            "Q+0.984": 129.04259353999998,
            "S+0.984": 88.016044435,
        }

    def _init_replacements(self) -> List[Tuple[str, str]]:
         return [
            ("C[Carbamidomethyl]", "C[UNIMOD:4]"),
            # Amino acid modifications.
            ("M[Oxidation]", "M[UNIMOD:35]"),  # Met oxidation
            ("N[Deamidated]", "N[UNIMOD:7]"),  # Asn deamidation
            ("Q[Deamidated]", "Q[UNIMOD:7]"),  # Gln deamidation
            ("K[TMT6plex]", "K[UNIMOD:737]"),
            ("S[Phospho]", "S[UNIMOD:21]"),
            ("T[Phospho]", "T[UNIMOD:21]"),
            ("Y[Phospho]", "Y[UNIMOD:21]"),
            # N-terminal modifications.
            ("[Acetyl]", "[UNIMOD:1]"),  # Acetylation
            ("[Carbamyl]", "[UNIMOD:5]"),  # Carbamylation
            ("[Ammonia-loss]", "[UNIMOD:385]"),  # NH3 loss
            ("[TMT6plex]", "[UNIMOD:737]"),
        ]

    def _add_n_term_token_score(self, scores: str) -> str:
        if not scores: return scores
        first_token_score = scores.split(",", 1)[0]
        scores = first_token_score + "," + scores
        return scores

    def format_sequence_and_scores(self, sequence: str, aa_scores: str):
        # direct (token-to-token) replacements
        for repl_args in self.replacements:
            sequence = sequence.replace(*repl_args)

        # format sequence and scores for n-term modifications
        if self.n_term_mod_pattern.search(sequence):
            aa_scores = self._add_n_term_token_score(aa_scores)

        return sequence, aa_scores

    def format_psms(self, psms: pd.DataFrame) -> pd.DataFrame:
        if psms.empty:
            return psms
            
        psms[["sequence", "aa_scores"]] = psms.apply(
            lambda row: self.format_sequence_and_scores(
                row["sequence"], row["aa_scores"]
            ),
            axis=1,
            result_type="expand",
        )
        return psms

    def read_casanovo_predictions(self, file_name: str) -> pd.DataFrame:
        psms = mztab.MzTab(file_name)["PSM"]
        psms.index = (
            psms["spectra_ref"]
            .str.extract(r"ms_run\[1\]:index=(\d+)")[0]
            .astype(int)
        )
        psms = psms[
            ["sequence", "search_engine_score[1]", "opt_ms_run[1]_aa_scores"]
        ]
        psms = psms.rename(
            {
                "search_engine_score[1]": "score",
                "opt_ms_run[1]_aa_scores": "aa_scores",
            },
            axis=1,
        )

        psms = self.format_psms(psms)

        invalid_ptms = (
            psms["sequence"]
            .str.findall(r"\[(?!UNIMOD:\d+\])[^\]]+\]")
            .explode()
            .dropna()
            .unique()
        )
        if len(invalid_ptms) > 0:
            print(f"Got unformatted rows, add the PTM to the REPLACEMENTS dict")
            print(invalid_ptms)

        return psms

    def parse_scores(self, aa_scores: str) -> list[float]:
        if not aa_scores:
            return []
        aa_scores = aa_scores.split(",")
        aa_scores = list(map(float, aa_scores))
        return aa_scores

    def format_scores(self, aa_scores: list[float]) -> str:
        return ",".join(map(str, aa_scores))

    def merge_n_term_score(self, aa_scores: str) -> str:
        aa_scores = self.parse_scores(aa_scores)
        if len(aa_scores) >= 2:
            aa_scores[0:2] = [np.mean(aa_scores[0:2])]
        return self.format_scores(aa_scores)

    def _transform_match_ptm(self, match: re.Match) -> str:
        ptm_idx = int(match.group(1))

        if ptm_idx not in self.ptm_masses:
            if self.unimod_db:
                try:
                    self.ptm_masses[ptm_idx] = self.unimod_db.get(ptm_idx).monoisotopic_mass
                except:
                    try:
                        self.ptm_masses[ptm_idx] = self.unimod_db.by_id(ptm_idx)["mono_mass"]
                    except:
                        # Fallback or error?
                        pass
        
        ptm_mass = self.ptm_masses.get(ptm_idx, 0.0) # Default to 0 if not found
        return f"[{ptm_mass:+}]"

    def _transform_match_n_term(self, match: re.Match) -> str:
        n_term_mod = match.group(1)
        first_aa = match.group(2)

        if match.group(3) is not None:
            first_aa_ptm = match.group(3)
            first_aa_ptm_mass = float(first_aa_ptm)
            n_term_mod_mass = float(n_term_mod)
            first_aa_ptm_mass += n_term_mod_mass
            first_aa_ptm = f"{first_aa_ptm_mass:+}"
        else:
            first_aa_ptm = n_term_mod
        return f"{first_aa}[{first_aa_ptm}]"

    def ptms_to_delta_mass(self, sequence, aa_scores):
        """Convert PTM representation from Unimod notation to delta mass notation."""
        sequence = self.ptm_pattern.sub(self._transform_match_ptm, sequence)
        sequence, merged_n_term_mod = self.n_term_pattern_proforma.subn(
            self._transform_match_n_term, sequence
        )

        if merged_n_term_mod == 1:
            aa_scores = self.merge_n_term_score(aa_scores)

        return sequence, aa_scores

    def merge_with_truth(self, pred_df, true_df):
        pred_df = true_df.merge(
            pred_df, how="left", left_index=True, right_index=True
        )
        return pred_df

    def calculate_metrics(self, output_data: pd.DataFrame) -> Dict[str, Any]:
        print(
            f"No predictions for {output_data['score'].isnull().sum()}/{len(output_data)} spectra"
        )
        
        output_data = output_data.sort_values("score", ascending=False)
        
        if len(output_data) == 0:
             return {
                "N sequences": 0,
                "N predicted": 0,
                "AA precision": 0.0,
                "AA recall": 0.0,
                "Pep precision": 0.0,
                "aa_matches_batch": [],
                "aa_scores": [],
            }

        labeled_idx = output_data["sequence_true"].notnull()
        not_sequenced_idx = output_data["sequence"].isnull()

        # Replace the sequences that were not sequenced with an empty string
        output_data["sequence"] = output_data["sequence"].fillna("")
        output_data["aa_scores"] = output_data["aa_scores"].fillna("")
        output_data["score"] = output_data["score"].fillna(0)
        
        # Apply ptms_to_delta_mass
        formatted_data = output_data.apply(
            lambda row: self.ptms_to_delta_mass(row["sequence"], row["aa_scores"]),
            axis=1,
            result_type="expand",
        )
        output_data["sequence"] = formatted_data[0]
        output_data["aa_scores"] = formatted_data[1]

        # Use library functions
        aa_matches_batch_list, n_aa1, n_aa2 = aa_match_batch(
            output_data["sequence"],
            output_data["sequence_true"],
            self.dynamic_mass_dict,
        )
        
        aa_precision, aa_recall, pep_precision = aa_match_metrics(
            aa_matches_batch_list, n_aa1, n_aa2
        )

        return {
            "N sequences": labeled_idx.sum(),
            "N predicted": labeled_idx.sum() - not_sequenced_idx.sum(),
            "AA precision": aa_precision,
            "AA recall": aa_recall,
            "Pep precision": pep_precision,
            "aa_matches_batch": aa_matches_batch_list,
            "aa_scores": output_data["aa_scores"][labeled_idx],
        }

    def evaluate_run(self, experiment_dir, run, val_true_df):
        mztab_file = os.path.join(experiment_dir, run, "validation.mztab")
        if not os.path.isfile(mztab_file):
            print(f"No mztab file found for run {run}")
            return None

        pred_df = self.read_casanovo_predictions(mztab_file)
        output_data = self.merge_with_truth(pred_df, val_true_df)
        return self.calculate_metrics(output_data)

    def evaluate_experiment(self, experiment_dir, val_true_df, cache_dir=None):
        run_metrics = {}
        for run in os.listdir(experiment_dir):
            # Skip non-directories if any, though os.listdir returns everything
            if not os.path.isdir(os.path.join(experiment_dir, run)):
                continue
                
            metrics_cache_file = None
            if cache_dir:
                metrics_cache_file = os.path.join(cache_dir, run, "metrics.p")
                if os.path.isfile(metrics_cache_file):
                    run_metrics[run] = pickle.load(open(metrics_cache_file, "rb"))
                    print(f"USING CACHED: {run}")
                    continue

            print(f"CALCULATING: {run}")
            metrics = self.evaluate_run(experiment_dir, run, val_true_df)
            run_metrics[run] = metrics

            if metrics is not None and metrics_cache_file:
                os.makedirs(os.path.join(cache_dir, run), exist_ok=True)
                pickle.dump(metrics, open(metrics_cache_file, "wb"))
                
        return run_metrics

# Top level functions for compatibility
def evaluate_experiment(experiment_dir, val_true_df, cache_dir=None):
    evaluator = ExperimentEvaluator()
    return evaluator.evaluate_experiment(experiment_dir, val_true_df, cache_dir)

def read_casanovo_predictions(file_name: str):
    evaluator = ExperimentEvaluator()
    return evaluator.read_casanovo_predictions(file_name)

def calculate_metrics(output_data):
    evaluator = ExperimentEvaluator()
    return evaluator.calculate_metrics(output_data)

def format_sequence_and_scores(sequence, aa_scores):
    evaluator = ExperimentEvaluator()
    return evaluator.format_sequence_and_scores(sequence, aa_scores)

# For testing exposure
_evaluator_instance = None
def get_evaluator():
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = ExperimentEvaluator()
    return _evaluator_instance
