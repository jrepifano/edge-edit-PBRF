import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_figure2(results, output_path="figure2.png"):
    """Plot predicted vs actual influence matching paper Figure 2 format.

    Creates a 2x3 grid: row (a) = deletions only, row (b) = insertions only.
    Reports Pearson correlation per panel.

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, (is_del_type, color, marker) in enumerate([
        (True, "red", "x"),
        (False, "blue", "o"),
    ]):
        for col, (metric, title) in enumerate(zip(metric_names, titles)):
            ax = axes[row, col]
            data = results[metric]
            predicted = np.array(data["predicted"])
            actual = np.array(data["actual"])
            is_del = np.array(data["is_deletion"])

            mask = is_del if is_del_type else ~is_del
            pred = predicted[mask]
            act = actual[mask]

            if len(pred) == 0:
                continue

            ax.scatter(pred, act, c=color, marker=marker, s=30, alpha=0.7)

            # Percentile-based axis limits (2nd-98th) for outlier robustness
            all_vals = np.concatenate([pred, act])
            vmin = np.percentile(all_vals, 2)
            vmax = np.percentile(all_vals, 98)
            margin = (vmax - vmin) * 0.05
            ax.set_xlim(vmin - margin, vmax + margin)
            ax.set_ylim(vmin - margin, vmax + margin)

            # Diagonal reference line (red dashed, matching paper)
            ax.plot(
                [vmin - margin, vmax + margin],
                [vmin - margin, vmax + margin],
                "r--", alpha=0.5, linewidth=1,
            )

            # Pearson correlation
            if len(pred) > 2:
                r_p, _ = pearsonr(pred, act)
                ax.text(
                    0.05, 0.95, f"Correlation: {r_p:.2f}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment="top",
                )

            ax.set_xlabel("Estimated Influence")
            ax.set_ylabel("Actual Influence")
            if row == 0:
                ax.set_title(title)

    # Row labels
    for row, label in enumerate(["(a) Deletion", "(b) Insertion"]):
        axes[row, 0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90, va="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {output_path}")
