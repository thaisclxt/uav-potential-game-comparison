import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from typing import Optional


def generate_uav_contribution_boxplots(outputs_dir: Path) -> None:
    """
    For each scenario folder like:
      outputs/UAVs{m}_GRID{grid}/

    Read the Greedy revenue workbook:
      revenue/UAVs{m}_GRID{grid}_Greedy.xlsx

    For each UAV j, compute its contribution share in each simulation run:
      share_j = (UAVj revenue at final round) / (sum_k UAVk revenue at final round)

    Then build ONE boxplot per scenario showing the distribution of share_j
    across all simulation runs, with x-axis labels:
      UAV0, UAV1, ..., UAV{m-1}

    Save the figure as:
      visualizations/uav_contribution_greedy.png
    """
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

        # One Greedy revenue file per scenario
        revenue_files = sorted(revenue_dir.glob("UAVs*_GRID*_Greedy.xlsx"))
        if not revenue_files:
            print(f"[BOX] No Greedy revenue files found in {revenue_dir}")
            continue

        # If there are multiple, we'll just use the first (your structure likely has one)
        revenue_file = revenue_files[0]

        try:
            sheets = pd.read_excel(revenue_file, sheet_name=None, index_col=0)
        except Exception as exc:
            print(f"[BOX] Could not read {revenue_file}: {exc}")
            continue

        # Collect per-UAV contribution shares over all simulation runs
        per_uav_shares: list[list[float]] = []
        uav_labels: list[str] = []

        # We will infer the UAV columns once from the first valid sheet
        first_uav_cols: Optional[list[str]] = None

        for run_name, df in sheets.items():
            uav_cols = [c for c in df.columns if str(c).upper().startswith("UAV")]
            if not uav_cols:
                print(f"[BOX] Skipping sheet {run_name}: no UAV columns found")
                continue

            if first_uav_cols is None:
                first_uav_cols = uav_cols
                uav_labels = [str(c) for c in uav_cols]
                per_uav_shares = [[] for _ in uav_cols]
            else:
                # Ensure consistent UAV columns across sheets
                if uav_cols != first_uav_cols:
                    print(
                        f"[BOX] Skipping sheet {run_name}: UAV columns {uav_cols} "
                        f"do not match first sheet {first_uav_cols}"
                    )
                    continue

            # Use final row (last index) as "final" revenue rate
            final_row = df[uav_cols].iloc[-1]
            total = float(final_row.sum())
            if total <= 0:
                print(
                    f"[BOX] Skipping sheet {run_name}: non-positive total revenue "
                    f"(total={total:.4f})"
                )
                continue

            # Compute shares for this run and append
            for idx, col in enumerate(uav_cols):
                share = float(final_row[col]) / total
                per_uav_shares[idx].append(share)

        if not per_uav_shares or all(len(v) == 0 for v in per_uav_shares):
            print(f"[BOX] No valid UAV contribution data found in {revenue_file.name}")
            continue

        visualizations_dir.mkdir(parents=True, exist_ok=True)
        out_path = visualizations_dir / "revenue_rate.png"

        # Create boxplot: one box per UAV, showing distribution over runs
        fig, ax = plt.subplots(figsize=(5, 6))

        bp = ax.boxplot(
            per_uav_shares,
            tick_labels=uav_labels,
            patch_artist=True,
        )

        # Style boxes
        for box in bp["boxes"]:
            box.set_facecolor("C0")
            box.set_edgecolor("black")

        for median in bp["medians"]:
            median.set(color="orange", linewidth=2)

        ax.set_ylabel("Share of total revenue rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[BOX] Saved UAV contribution boxplot to {out_path}")


def _algo_label_from_seq_file(seq_file: Path) -> Optional[str]:
    """
    Infer algorithm label from the sequence file name.

    Examples:
      UAVs3_GRID10_IRADA_sequences.xlsx       -> IRADA
      UAVs3_GRID10_ModeX_Y_sequences.xlsx     -> ModeX_Y
    """
    stem = seq_file.stem.replace("_sequences", "")

    if "IRADA" in stem:
        return "IRADA"

    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.startswith("Mode"):
            if i + 1 < len(parts):
                return f"{p}_{parts[i + 1]}"
            return p

    return None


def get_revenue_file_for_sequence(seq_file: Path, rev_dir: Path) -> Path | None:
    """
    Given a sequence file, find the matching revenue Excel file in rev_dir.

    Matching is based on:
      - UAVs{m}_GRID{grid_size} prefix
      - optional algorithm label if present in filename
    """
    stem = seq_file.stem.replace("_sequences", "")
    parts = stem.split("_")

    if len(parts) < 2:
        return None

    prefix = f"{parts[0]}_{parts[1]}"

    matches = sorted(rev_dir.glob(f"{prefix}_*.xlsx"))
    if not matches:
        matches = sorted(rev_dir.glob(f"{prefix}.xlsx"))

    if not matches:
        return None

    seq_algo = (_algo_label_from_seq_file(seq_file) or "").lower()

    if seq_algo:
        for candidate in matches:
            if seq_algo in candidate.stem.lower():
                return candidate

    return matches[0]
