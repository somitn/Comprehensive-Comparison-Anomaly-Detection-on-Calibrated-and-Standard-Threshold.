import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def fig_splitting():
    # ------------------------------------------------------------------
    # Row counts
    # ------------------------------------------------------------------
    NORMAL_TOTAL = 496_800
    ATTACK_TOTAL = 449_919

    TRAIN_ROWS      = 100_000
    VAL_ROWS        =  20_000
    NORMAL_UNUSED   = NORMAL_TOTAL - TRAIN_ROWS - VAL_ROWS    # 376,800

    TUNE_ROWS       =  89_983
    TEST_ROWS       = 359_936
    ATTACK_UNUSED   = ATTACK_TOTAL - TUNE_ROWS - TEST_ROWS    # 0

    # ------------------------------------------------------------------
    # Categories and colors
    # ------------------------------------------------------------------
    CATEGORIES = ["Train", "Validation", "Tune", "Test", "Unused"]
    COLORS = {
        "Train":      "#1f77b4",   # blue
        "Validation": "#aec7e8",   # light blue
        "Tune":       "#2ca02c",   # green
        "Test":       "#d62728",   # red
        "Unused":     "#d9d9d9",   # light gray
    }

    # ------------------------------------------------------------------
    # Fractions per row (each row must sum to 1.0)
    # ------------------------------------------------------------------
    normal_fracs = {
        "Train":      TRAIN_ROWS    / NORMAL_TOTAL,
        "Validation": VAL_ROWS      / NORMAL_TOTAL,
        "Tune":       0.0,
        "Test":       0.0,
        "Unused":     NORMAL_UNUSED / NORMAL_TOTAL,
    }
    attack_fracs = {
        "Train":      0.0,
        "Validation": 0.0,
        "Tune":       TUNE_ROWS     / ATTACK_TOTAL,
        "Test":       TEST_ROWS     / ATTACK_TOTAL,
        "Unused":     ATTACK_UNUSED / ATTACK_TOTAL,
    }

    assert abs(sum(normal_fracs.values()) - 1.0) < 1e-6, \
        f"Normal fractions sum to {sum(normal_fracs.values())}"
    assert abs(sum(attack_fracs.values()) - 1.0) < 1e-6, \
        f"Attack fractions sum to {sum(attack_fracs.values())}"

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4.0))

    # Each row: (label, y_position, fractions, row_counts_for_annotation)
    rows = [
        (
            "NORMAL FILE\n496,800 rows\n(all normal, no labels)",
            0,                         # y-position = 0 → bottom
            normal_fracs,
            {"Train": TRAIN_ROWS, "Validation": VAL_ROWS,
             "Unused": NORMAL_UNUSED},
        ),
        (
            "ATTACK FILE\n449,919 rows\n(mixed labels)",
            1,                         # y-position = 1 → top
            attack_fracs,
            {"Tune": TUNE_ROWS, "Test": TEST_ROWS},
        ),
    ]

    # ------------------------------------------------------------------
    # Draw bars
    # ------------------------------------------------------------------
    for label, y_pos, fracs, row_counts in rows:
        left = 0.0
        for cat in CATEGORIES:
            f = fracs[cat]
            if f > 0.03:
                # Wide slice: label inside the bar
                ax.barh(y_pos, f, left=left, height=0.55,
                        color=COLORS[cat],
                        edgecolor="black", linewidth=0.8)
                ax.text(
                    left + f / 2, y_pos,
                    f"{cat}\n{row_counts[cat]:,}\n({f:.1%})",
                    ha="center", va="center",
                    color="white" if cat != "Unused" else "black",
                    fontsize=9, fontweight="bold",
                )
            elif f > 0:
                # Thin slice: just draw the bar (no label)
                ax.barh(y_pos, f, left=left, height=0.55,
                        color=COLORS[cat],
                        edgecolor="black", linewidth=0.8)
            left += f

    # ------------------------------------------------------------------
    # Axes and labels
    # ------------------------------------------------------------------
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [rows[0][0], rows[1][0]],
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel(
        "Fraction of source file (time-ordered split, no shuffling)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title(
        "Dataset Splitting — Normal file and Attack file",
        fontsize=12, pad=12,
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    handles = [
        Patch(facecolor=COLORS[c], edgecolor="black", label=c)
        for c in CATEGORIES
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    # ------------------------------------------------------------------
    # Save and show
    # ------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig("fig2_splitting.png", dpi=150, bbox_inches="tight")
    plt.show()

    return fig


if __name__ == "__main__":
    fig_splitting()