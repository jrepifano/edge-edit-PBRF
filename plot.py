import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_figure2(results, output_path="figure2.png"):
    """Plot 1x3 scatter plots of predicted vs actual influence.

    Args:
        results: dict with keys 'validation_loss', 'over_squashing', 'dirichlet_energy'
            each mapping to a dict with:
                'predicted': list of floats
                'actual': list of floats
                'is_deletion': list of bools
        output_path: path to save the figure
    """
    metric_names = ["validation_loss", "over_squashing", "dirichlet_energy"]
    titles = ["Validation Loss", "Over-squashing", "Dirichlet Energy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (metric, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx]
        data = results[metric]
        predicted = np.array(data["predicted"])
        actual = np.array(data["actual"])
        is_del = np.array(data["is_deletion"])

        # Plot deletions as red x, insertions as blue o
        del_mask = is_del
        ins_mask = ~is_del

        if del_mask.sum() > 0:
            ax.scatter(
                predicted[del_mask],
                actual[del_mask],
                c="red",
                marker="x",
                s=30,
                label="Deletion",
                alpha=0.7,
            )
        if ins_mask.sum() > 0:
            ax.scatter(
                predicted[ins_mask],
                actual[ins_mask],
                c="blue",
                marker="o",
                s=20,
                label="Insertion",
                alpha=0.7,
            )

        # Diagonal line
        all_vals = np.concatenate([predicted, actual])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05
        ax.plot(
            [vmin - margin, vmax + margin],
            [vmin - margin, vmax + margin],
            "k--",
            alpha=0.5,
            linewidth=1,
        )

        # Correlation
        if len(predicted) > 2:
            corr, _ = pearsonr(predicted, actual)
            ax.text(
                0.05,
                0.95,
                f"Correlation: {corr:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

        ax.set_xlabel("Estimated Influence")
        ax.set_ylabel("Actual Influence")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {output_path}")
