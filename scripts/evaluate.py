import os
import pickle
import re
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from pyteomics import mztab

# Instance of Unimod database to look up PTM masses
try:
    from pyteomics.mass.unimod import Unimod

    UNIMOD_DB = Unimod()
except:
    from pyteomics.mass import Unimod

    UNIMOD_DB = Unimod()
# Cache of PTM masses to avoid repeated lookups in the Unimod database
PTM_MASSES = {}

AA_MASSES = {
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
}


REPLACEMENTS = [
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
    # ("+43.006-17.027", "[UNIMOD:5][UNIMOD:385]"),   # Carbamylation and NH3 loss
]
PEP_SPLIT_PATTERN = r"(?<=.)(?=[A-Z])"
N_TERM_MOD_PATTERN = r"^((\[UNIMOD:[0-9]+\])+)"


def _add_n_term_token_score(scores: str) -> str:
    first_token_score = scores.split(",", 1)[0]
    scores = first_token_score + "," + scores
    return scores


def format_sequence_and_scores(sequence, aa_scores):
    # direct (token-to-token) replacements
    for repl_args in REPLACEMENTS:
        sequence = sequence.replace(*repl_args)

    # format sequence and scores for n-term modifications
    if re.search(N_TERM_MOD_PATTERN, sequence):
        aa_scores = _add_n_term_token_score(aa_scores)

    return sequence, aa_scores


def format_psms(psms):
    psms[["sequence", "aa_scores"]] = psms.apply(
        lambda row: format_sequence_and_scores(
            row["sequence"], row["aa_scores"]
        ),
        axis=1,
        result_type="expand",
    )
    return psms


def read_casanovo_predictions(file_name: str) -> pd.DataFrame:
    """
    Read casanovo generated mztab file and format the sequences and scores

    Parameters
    ----------
    file_name

    Returns
    -------
    Dataframe with psms
    """
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

    psms = format_psms(psms)

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


def parse_scores(aa_scores: str) -> list[float]:
    """
    TODO.
    * assumes that AA confidence scores always come
    as a string of float numbers separated by a comma.
    """
    if not aa_scores:
        return []
    aa_scores = aa_scores.split(",")
    aa_scores = list(map(float, aa_scores))
    return aa_scores


def format_scores(aa_scores: list[float]) -> str:
    """
    Write a list of float per-token scores
    into a string of float scores separated by ','.
    """
    return ",".join(map(str, aa_scores))


def merge_n_term_score(aa_scores: str) -> str:
    aa_scores = parse_scores(aa_scores)
    aa_scores[0:2] = [np.mean(aa_scores[0:2])]
    return format_scores(aa_scores)


def _transform_match_ptm(match: re.Match) -> str:
    """
    TODO
    """
    ptm_idx = int(match.group(1))

    if ptm_idx not in PTM_MASSES:
        try:
            PTM_MASSES[ptm_idx] = UNIMOD_DB.get(ptm_idx).monoisotopic_mass
        except:
            PTM_MASSES[ptm_idx] = UNIMOD_DB.by_id(ptm_idx)["mono_mass"]

    ptm_mass = PTM_MASSES[ptm_idx]
    return f"[{ptm_mass:+}]"


def _transform_match_n_term(match: re.Match) -> str:
    """
    TODO
    """
    n_term_mod = match.group(1)
    first_aa = match.group(2)

    if match.group(3) is not None:
        first_aa_ptm = match.group(3)
        first_aa_ptm_mass = float(first_aa_ptm)  # convert from str to float
        n_term_mod_mass = float(
            n_term_mod
        )  # convert n_term_mod_mass from str to float
        first_aa_ptm_mass += n_term_mod_mass  # sum up
        first_aa_ptm = f"{first_aa_ptm_mass:+}"  # convert back to string
    else:
        first_aa_ptm = n_term_mod
    return f"{first_aa}[{first_aa_ptm}]"


def ptms_to_delta_mass(sequence, aa_scores):
    """Convert PTM representation from Unimod notation to delta mass notation."""

    PTM_PATTERN = r"\[UNIMOD:([0-9]+)\]"  # find ptms
    N_TERM_PATTERN = r"^\[([0-9+-.]+)\]-([A-Z])(?:\[([0-9+-.]+)\])?"  # find n-term modifications in ProForma notation

    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)

    sequence, merged_n_term_mod = re.subn(
        N_TERM_PATTERN, _transform_match_n_term, sequence
    )

    if merged_n_term_mod == 1:
        aa_scores = merge_n_term_score(aa_scores)

    return sequence, aa_scores


def merge_with_truth(pred_df, true_df):
    pred_df = true_df.merge(
        pred_df, how="left", left_index=True, right_index=True
    )
    return pred_df


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match, (aa_matches_1, aa_matches_2) = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return (
            aa_matches[::-1],
            pep_match,
            (aa_matches_1[::-1], aa_matches_2[::-1]),
        )
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool, Tuple[np.ndarray]]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match,
        (iii) TODO.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = (
                re.split(r"(?<=.)(?=[A-Z])", peptide1) if peptide1 else []
            )
        if isinstance(peptide2, str):
            peptide2 = (
                re.split(r"(?<=.)(?=[A-Z])", peptide2) if peptide2 else []
            )
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return float(aa_precision), float(aa_recall), float(pep_precision)


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    aa_matches_1 = np.zeros(len(peptide1), np.bool_)
    aa_matches_2 = np.zeros(len(peptide2), np.bool_)

    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = get_token_mass(peptide1[i1], aa_dict, 0)
        aa_mass2 = get_token_mass(peptide2[i2], aa_dict, 0)

        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            match = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            aa_matches[max(i1, i2)] = match
            aa_matches_1[i1] = match
            aa_matches_2[i2] = match

            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2

        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all(), (aa_matches_1, aa_matches_2)


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match, (aa_matches_1, aa_matches_2) = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match, (aa_matches_1, aa_matches_2)

    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = get_token_mass(peptide1[i1], aa_dict, 0)
        aa_mass2 = get_token_mass(peptide2[i2], aa_dict, 0)

        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            match = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            aa_matches[max(i1, i2)] = match
            aa_matches_1[i1] = match
            aa_matches_2[i2] = match

            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2

        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all(), (aa_matches_1, aa_matches_2)


# Method is borrowed from 'spectrum_utils' package
# (temporary removed 'spectrum_utils' dependence
# due to numba cache issues when running on HPC cluster).
def mass_diff(mz1, mz2, mode_is_da):
    """
    Calculate the mass difference(s).

    Parameters
    ----------
    mz1
        First m/z value(s).
    mz2
        Second m/z value(s).
    mode_is_da : bool
        Mass difference in Dalton (True) or in ppm (False).

    Returns
    -------
        The mass difference(s) between the given m/z values.
    """
    return mz1 - mz2 if mode_is_da else (mz1 - mz2) / mz2 * 10**6


def get_token_mass(
    token: str, aa_dict: Dict[str, float], default: float = 0
) -> float:
    """TODO."""

    def safe_float(s):
        try:
            return float(s)
        except ValueError:
            return default

    if len(token) == 1:
        mass = aa_dict.get(token, default)
    else:
        aa, ptm = token[0], token[2:-1]  # not a transparent way
        aa_mass = aa_dict.get(aa, default)
        ptm_mass = safe_float(ptm)  # float(ptm)
        mass = aa_mass + ptm_mass
    return mass


def calculate_metrics(output_data):
    print(
        f"No predictions for {output_data['score'].isnull().sum()}/{len(output_data)} spectra"
    )

    output_data = output_data.sort_values("score", ascending=False)
    labeled_idx = output_data["sequence_true"].notnull()
    not_sequenced_idx = output_data["sequence"].isnull()

    # Replace the sequences that were not sequenced with an empty string
    output_data["sequence"] = output_data["sequence"].fillna("")
    output_data["aa_scores"] = output_data["aa_scores"].fillna("")
    output_data["score"] = output_data["score"].fillna(0)

    output_data[["sequence", "aa_scores"]] = output_data.apply(
        lambda row: ptms_to_delta_mass(row["sequence"], row["aa_scores"]),
        axis=1,
        result_type="expand",
    ).values

    aa_matches_batch, n_aa1, n_aa2 = aa_match_batch(
        output_data["sequence"],
        output_data["sequence_true"],
        AA_MASSES,
    )
    aa_precision, aa_recall, pep_precision = aa_match_metrics(
        aa_matches_batch, n_aa1, n_aa2
    )

    return {
        "N sequences": labeled_idx.sum(),
        "N predicted": labeled_idx.sum() - not_sequenced_idx.sum(),
        "AA precision": aa_precision,
        "AA recall": aa_recall,
        "Pep precision": pep_precision,
        "aa_matches_batch": aa_matches_batch,
        "aa_scores": output_data["aa_scores"][labeled_idx],
    }


def evaluate_run(experiment_dir, run, val_true_df):
    mztab_file = os.path.join(experiment_dir, run, "validation.mztab")
    if not os.path.isfile(mztab_file):
        print(f"No mztab file found for run {run}")
        return None

    pred_df = read_casanovo_predictions(mztab_file)

    output_data = merge_with_truth(pred_df, val_true_df)

    return calculate_metrics(output_data)


def evaluate_experiment(experiment_dir, val_true_df, cache_dir=None):
    run_metrics = {}
    for run in os.listdir(experiment_dir):
        metrics_cache_file = os.path.join(cache_dir, run, "metrics.p")
        if cache_dir is not None and os.path.isfile(metrics_cache_file):
            run_metrics[run] = pickle.load(open(metrics_cache_file, "rb"))
            print(f"USING CACHED: {run}")
            continue

        print(f"CALCULATING: {run}")
        metrics = evaluate_run(experiment_dir, run, val_true_df)
        run_metrics[run] = metrics

        if metrics is not None and cache_dir is not None:
            os.makedirs(os.path.join(cache_dir, run), exist_ok=True)
            pickle.dump(metrics, open(metrics_cache_file, "wb"))
    return run_metrics
