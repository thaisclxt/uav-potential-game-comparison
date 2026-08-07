import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional
from matplotlib.ticker import PercentFormatter


FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 24

def set_plot_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
    })


def generate_uav_contribution_boxplots(outputs_dir: Path) -> None:
    """
    For each scenario folder:
      results/algorithm/UAVs{m}_GRID{grid}/

    Read every revenue workbook in:
      revenue/*.xlsx

    For each workbook (algorithm/configuration), compute for each UAV j:
      share_j = (UAVj revenue at final round) / (sum_k UAVk revenue at final round)

    Then build ONE boxplot per workbook showing the distribution of share_j
    across all runs, with x-axis labels:
      UAV0, UAV1, ..., UAV{m-1}

    Save as:
      visualizations/uav_contribution_<algo>.png
    """
    set_plot_style()
    
    outputs_dir = Path(outputs_dir)

    if not outputs_dir.exists():
        print(f"[BOX] Output directory does not exist: {outputs_dir}")
        return

    scenario_dirs = sorted(
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.startswith("UAVs")
    )

    if not scenario_dirs:
        print(f"[BOX] No scenario folders found in {outputs_dir}")
        return

    for scenario_dir in scenario_dirs:
        revenue_dir = scenario_dir / "revenue"
        visualizations_dir = scenario_dir / "visualizations"

        if not revenue_dir.exists():
            print(f"[BOX] Missing revenue dir: {revenue_dir}")
            continue

        revenue_files = sorted(
            f for f in revenue_dir.glob("*.xlsx")
            if not f.stem.endswith("_stats")
        )

        if not revenue_files:
            print(f"[BOX] No revenue files found in {revenue_dir}")
            continue

        visualizations_dir.mkdir(parents=True, exist_ok=True)

        for revenue_file in revenue_files:
            try:
                sheets = pd.read_excel(revenue_file, sheet_name=None, index_col=0)
            except Exception as exc:
                print(f"[BOX] Could not read {revenue_file}: {exc}")
                continue

            per_uav_shares: list[list[float]] = []
            uav_labels: list[str] = []
            first_uav_cols: Optional[list[str]] = None

            for run_name, df in sheets.items():
                uav_cols = [c for c in df.columns if str(c).upper().startswith("UAV")]
                if not uav_cols:
                    print(f"[BOX] Skipping sheet {run_name} in {revenue_file.name}: no UAV columns found")
                    continue

                if first_uav_cols is None:
                    first_uav_cols = uav_cols
                    uav_labels = [str(c) for c in uav_cols]
                    per_uav_shares = [[] for _ in uav_cols]
                elif uav_cols != first_uav_cols:
                    print(
                        f"[BOX] Skipping sheet {run_name} in {revenue_file.name}: "
                        f"UAV columns {uav_cols} do not match first sheet {first_uav_cols}"
                    )
                    continue

                final_row = df[uav_cols].iloc[-1]
                total = float(final_row.sum())
                if total <= 0:
                    print(
                        f"[BOX] Skipping sheet {run_name} in {revenue_file.name}: "
                        f"non-positive total revenue (total={total:.4f})"
                    )
                    continue

                for idx, col in enumerate(uav_cols):
                    share = float(final_row[col]) / total
                    per_uav_shares[idx].append(share)

            if not per_uav_shares or all(len(v) == 0 for v in per_uav_shares):
                print(f"[BOX] No valid UAV contribution data found in {revenue_file.name}")
                continue

            algo_label = revenue_file.stem.replace(f"{scenario_dir.name}_", "")
            safe_algo = algo_label.replace(" ", "_").replace("/", "_")
            out_path = visualizations_dir / f"uav_contribution_{safe_algo}.png"

            fig, ax = plt.subplots(figsize=(1.0 * len(uav_cols) + 3, 6))
            bp = ax.boxplot(per_uav_shares, tick_labels=uav_cols, patch_artist=True)

            for box in bp["boxes"]:
                box.set_facecolor("C0")
                box.set_edgecolor("black")

            for median in bp["medians"]:
                median.set(color="orange", linewidth=2)

            ax.set_ylabel("Share of total revenue rate")

            all_vals = np.concatenate(
                [np.asarray(d, float) for d in per_uav_shares if len(d)]
            )
            if all_vals.size > 0:
                max_share = float(all_vals.max())     # e.g. 0.132
                if max_share <= 0:
                    top_frac = 0.1
                else:
                    # nearest 10% ceiling, capped at 100%
                    top_frac = min(1.0, math.ceil(max_share * 10) / 10.0)
            else:
                top_frac = 1.0

            ax.set_ylim(0, top_frac)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

            step = 0.05 if top_frac <= 0.5 else 0.1
            ax.set_yticks(np.arange(0.0, top_frac + 1e-9, step))

            plt.xticks(rotation=0, ha="center")
            plt.tight_layout()


            # ax.grid(True, axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

            print(f"[BOX] Saved UAV contribution boxplot to {out_path}")
