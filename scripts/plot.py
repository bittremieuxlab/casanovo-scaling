import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc

def parse_scores(aa_scores: str) -> list[float]:
    """
    assumes that AA confidence scores always come
    as a string of float numbers separated by a comma.
    """
    if aa_scores == "":
        return []
    aa_scores = aa_scores.split(",")
    aa_scores = list(map(float, aa_scores))
    return aa_scores


def plot_precision_coverage_curve(data, title: str, save_file, show=False):
    precision = np.cumsum(data) / np.arange(1, len(data) + 1)
    coverage = np.arange(1, len(data) + 1) / len(data)
    plot_idxs = np.linspace(0, len(coverage) - 1, 100).astype(np.int64)
    pc_auc = auc(coverage, precision)
    plt.plot(
        coverage[plot_idxs],
        precision[plot_idxs],
        label=f"AUC = {pc_auc:.3f}",
    )
    plt.ylim(0, 1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().set_aspect("equal")

    plt.title(title)
    plt.xlabel("coverage")
    plt.ylabel("precision")
    plt.legend()
    plt.savefig(save_file, dpi=300)
    if show:
        plt.show()
    plt.clf()
    return pc_auc

def plot_metrics(metrics, name, save_name, show=False):
    precision_data = np.array([aa_match[1] for aa_match in metrics["aa_matches_batch"]])
    peptide_pc_auc = plot_precision_coverage_curve(
        precision_data,
        f"{name}\nPeptide Precision-Coverage",
        f"{save_name}_peptide.png",
        show,
    )

    # For amino acids
    aa_scores = np.concatenate(
        list(map(parse_scores, metrics["aa_scores"].values.tolist()))
    )
    sorted_idx = np.argsort(aa_scores)[::-1]
    aa_match_data = np.concatenate(
        [aa_match[2][0] for aa_match in metrics["aa_matches_batch"]]
    )

    aa_pc_auc = plot_precision_coverage_curve(
        aa_match_data[sorted_idx],
        f"{name}\nAmino Acid Precision-Coverage",
        f"{save_name}_aa.png",
        show,
    )
    return peptide_pc_auc, aa_pc_auc