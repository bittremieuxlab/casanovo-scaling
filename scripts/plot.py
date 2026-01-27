import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def get_min_valid_loss(experiment_dir, run):
    """
    Read the metrics.csv file from a run's directory and return
    the minimum validation loss achieved during training.
    """
    metrics_csv_path = os.path.join(experiment_dir, run, "csv_logs", "metrics.csv")
    if not os.path.isfile(metrics_csv_path):
        return None
    
    df = pd.read_csv(metrics_csv_path)
    valid_loss = df["valid_CELoss"].dropna()
    if len(valid_loss) == 0:
        return None
    return valid_loss.min()


def plot_precision_vs_loss(
    run_metrics: dict,
    experiment_dir: str,
    save_file: str = None,
    show: bool = False,
):
    """
    Plot peptide and AA precision against validation loss for multiple runs.
    
    Args:
        run_metrics: Dictionary mapping run names to their metric dictionaries.
                     Each metric dict should have 'Pep precision' and 'AA precision'.
        experiment_dir: Path to the experiment directory containing run subdirectories.
        save_file: Optional path to save the figure (without extension; will save as PNG).
        show: If True, display the plot.
    
    Returns:
        Tuple of (losses, pep_precisions, aa_precisions) for further analysis.
    """
    losses = []
    pep_precisions = []
    aa_precisions = []
    run_names = []
    
    for run, metrics in run_metrics.items():
        if metrics is None:
            continue
        
        min_loss = get_min_valid_loss(experiment_dir, run)
        if min_loss is None:
            continue
        
        losses.append(min_loss)
        pep_precisions.append(metrics.get("Pep precision", 0.0))
        aa_precisions.append(metrics.get("AA precision", 0.0))
        run_names.append(run)
    
    if not losses:
        print("No data to plot - no runs with both metrics and valid loss found.")
        return [], [], []
    
    losses = np.array(losses)
    pep_precisions = np.array(pep_precisions)
    aa_precisions = np.array(aa_precisions)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot peptide precision vs loss
    ax1 = axes[0]
    ax1.scatter(losses, pep_precisions, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel("Minimum Validation Loss")
    ax1.set_ylabel("Peptide Precision")
    ax1.set_title("Peptide Precision vs Validation Loss")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Add correlation coefficient
    if len(losses) > 1:
        corr = np.corrcoef(losses, pep_precisions)[0, 1]
        ax1.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top')
    
    # Plot AA precision vs loss
    ax2 = axes[1]
    ax2.scatter(losses, aa_precisions, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel("Minimum Validation Loss")
    ax2.set_ylabel("AA Precision")
    ax2.set_title("AA Precision vs Validation Loss")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # Add correlation coefficient
    if len(losses) > 1:
        corr = np.corrcoef(losses, aa_precisions)[0, 1]
        ax2.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(f"{save_file}.png", dpi=300)
    if show:
        plt.show()
    plt.clf()
    
    return losses, pep_precisions, aa_precisions

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