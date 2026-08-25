from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# Plot style
# ============================================================

FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 24


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
    })


# ============================================================
# Workbook helpers
# ============================================================

def _uav_sort_key(column: str) -> tuple[int, str]:
    """Sort UAV0, UAV1, UAV2, ... numerically."""
    text = str(column)
    suffix = text.upper().replace("UAV", "")

    try:
        return int(suffix), text
    except ValueError:
        return 999999, text


def _matching_workbooks(
    root: Path,
    uav_count: int,
    workbook_pattern: str,
) -> list[Path]:
    """
    Find revenue workbooks for one UAV count.

    Expected workbook filename prefix:
        UAVs3_...
        UAVs4_...
        ...
    """
    if not root.exists():
        print(f"[WARN] Directory does not exist: {root}")
        return []

    workbooks: list[Path] = []

    for workbook in root.rglob(workbook_pattern):
        if not workbook.is_file():
            continue

        if not workbook.name.startswith(f"UAVs{uav_count}_"):
            continue

        stem_lower = workbook.stem.lower()

        if workbook.stem.endswith("_stats"):
            continue

        if "sequences" in stem_lower:
            continue

        workbooks.append(workbook)

    return sorted(workbooks)


def _final_total_revenue_rates(workbook: Path) -> list[float]:
    """
    Return one final total revenue-rate value per Excel sheet/run.

    Final total revenue rate =
        sum(UAV0, UAV1, ..., UAVn) in the final row.
    """
    try:
        with pd.ExcelFile(workbook) as excel_file:
            first_100_sheet_names = excel_file.sheet_names[:100]

            sheets = pd.read_excel(
                excel_file,
                sheet_name=first_100_sheet_names,
                index_col=0,
            )
    except Exception as exc:
        print(f"[WARN] Could not read {workbook}: {exc}")
        return []

    final_totals: list[float] = []

    for sheet_name, frame in sheets.items():
        uav_columns = sorted(
            [
                column
                for column in frame.columns
                if str(column).upper().startswith("UAV")
            ],
            key=_uav_sort_key,
        )

        if frame.empty or not uav_columns:
            print(
                f"[WARN] Skipping {workbook.name}:{sheet_name}; "
                f"no UAV columns or no rows."
            )
            continue

        numeric_frame = frame[uav_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        final_total = float(numeric_frame.iloc[-1].sum())

        if np.isfinite(final_total):
            final_totals.append(final_total)

    return final_totals


def _mode_label(workbook: Path, prefix: str) -> str:
    """
    Create Analysis.py-style labels.

    Examples:
        ..._ModeGG_Random.xlsx      -> NRGG / ORGG
        ..._ModeGR_Sequential.xlsx  -> NSGR / OSGR
    """
    parts = workbook.stem.split("_")

    mode_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.startswith("Mode")
        ),
        None,
    )

    if mode_index is None:
        return workbook.stem

    mode_suffix = parts[mode_index].replace("Mode", "")

    order_token = (
        parts[mode_index + 1].lower()
        if mode_index + 1 < len(parts)
        else ""
    )

    if order_token.startswith("random"):
        order_code = "R"
    elif order_token.startswith("sequential"):
        order_code = "S"
    else:
        order_code = "?"

    return f"{prefix}{order_code}{mode_suffix}"


# ============================================================
# Best configuration selection
# ============================================================

def _select_best_configuration(
    root: Path,
    uav_count: int,
    workbook_pattern: str,
    prefix: str,
) -> tuple[Path | None, str | None]:
    """
    Select the workbook with the largest mean final total revenue rate.

    Used independently for:
        - Non-overlap
        - Overlap
    """
    workbooks = _matching_workbooks(
        root=root,
        uav_count=uav_count,
        workbook_pattern=workbook_pattern,
    )

    if not workbooks:
        print(
            f"[WARN] No candidate workbooks for "
            f"UAVs={uav_count} in {root}"
        )
        return None, None

    best_workbook: Path | None = None
    best_mean_rate = float("-inf")

    for workbook in workbooks:
        final_rates = _final_total_revenue_rates(workbook)

        if not final_rates:
            continue

        mean_final_rate = float(np.mean(final_rates))

        if mean_final_rate > best_mean_rate:
            best_mean_rate = mean_final_rate
            best_workbook = workbook

    if best_workbook is None:
        print(
            f"[WARN] No usable candidate workbook for "
            f"UAVs={uav_count} in {root}"
        )
        return None, None

    label = _mode_label(best_workbook, prefix)

    print(
        f"[INFO] UAVs={uav_count}: selected {label}; "
        f"mean final total revenue rate={best_mean_rate:.4f}"
    )

    return best_workbook, label


# ============================================================
# Per-UAV revenue-share calculation
# ============================================================

def _per_uav_shares_from_workbooks(
    workbooks: list[Path],
) -> dict[str, list[float]]:
    """
    For every run/sheet:

    1. Find the negotiation round with the highest total revenue rate.
    2. Compute each UAV's contribution at that round:

        share_j = UAVj / sum(UAV0, UAV1, ..., UAVn)
    """
    shares_by_uav: dict[str, list[float]] = defaultdict(list)

    for workbook in workbooks:
        try:
            with pd.ExcelFile(workbook) as excel_file:
                first_100_sheet_names = excel_file.sheet_names[:100]

                sheets = pd.read_excel(
                    excel_file,
                    sheet_name=first_100_sheet_names,
                    index_col=0,
                )
        except Exception as exc:
            print(f"[WARN] Could not read {workbook}: {exc}")
            continue

        for sheet_name, frame in sheets.items():
            uav_columns = sorted(
                [
                    column
                    for column in frame.columns
                    if str(column).upper().startswith("UAV")
                ],
                key=_uav_sort_key,
            )

            if frame.empty or not uav_columns:
                print(
                    f"[WARN] Skipping {workbook.name}:{sheet_name}; "
                    f"no UAV data."
                )
                continue

            numeric_frame = frame[uav_columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).fillna(0.0)

            total_by_round = numeric_frame.sum(axis=1)

            if total_by_round.empty or (total_by_round <= 0.0).all():
                print(
                    f"[WARN] Skipping {workbook.name}:{sheet_name}; "
                    f"all total rates are non-positive."
                )
                continue

            best_round_index = total_by_round.idxmax()
            best_total = float(total_by_round.loc[best_round_index])

            if best_total <= 0.0:
                continue

            best_row = numeric_frame.loc[best_round_index]

            for column in uav_columns:
                share = float(best_row[column]) / best_total
                shares_by_uav[str(column)].append(share)

    return shares_by_uav


def _source_shares_for_uav_count(
    source_name: str,
    source: Mapping[str, object],
    uav_count: int,
) -> tuple[str, dict[str, list[float]]] | None:
    """
    Return:

        plot_label,
        {UAV0: [...], UAV1: [...], ...}

    Source kinds:
        best_nonoverlap
        best_overlap
        single
    """
    root = Path(source["root"])
    workbook_pattern = str(source.get("workbook_pattern", "*.xlsx"))
    source_kind = str(source.get("kind", "single")).lower()

    if source_kind == "best_nonoverlap":
        best_workbook, best_label = _select_best_configuration(
            root=root,
            uav_count=uav_count,
            workbook_pattern=workbook_pattern,
            prefix="N",
        )

        if best_workbook is None or best_label is None:
            return None

        shares_by_uav = _per_uav_shares_from_workbooks(
            [best_workbook]
        )

        return best_label, shares_by_uav

    if source_kind == "best_overlap":
        best_workbook, best_label = _select_best_configuration(
            root=root,
            uav_count=uav_count,
            workbook_pattern=workbook_pattern,
            prefix="O",
        )

        if best_workbook is None or best_label is None:
            return None

        shares_by_uav = _per_uav_shares_from_workbooks(
            [best_workbook]
        )

        return best_label, shares_by_uav

    workbooks = _matching_workbooks(
        root=root,
        uav_count=uav_count,
        workbook_pattern=workbook_pattern,
    )

    if not workbooks:
        print(
            f"[WARN] No workbook found for {source_name}, "
            f"UAVs={uav_count}, in {root}"
        )
        return None

    shares_by_uav = _per_uav_shares_from_workbooks(workbooks)

    return source_name, shares_by_uav


# ============================================================
# Plotting
# ============================================================

def _safe_filename_label(label: str) -> str:
    """Convert a label into a safe output filename token."""
    return (
        label.replace("+", "Plus")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _save_per_uav_share_boxplot(
    label: str,
    shares_by_uav: dict[str, list[float]],
    uav_count: int,
    output_directory: Path,
    show_title: bool = True,
) -> Path | None:
    """
    Save one per-UAV revenue-share boxplot.

    Styling matches Analysis.py::boxplot_uav_contribution_all():
        - Blue C0 boxes
        - Black edges
        - Orange medians
        - Percentage y-axis
        - No grid
        - UAV0, UAV1, ... labels
    """
    uav_labels = [
        f"UAV{index}"
        for index in range(uav_count)
    ]

    missing_uavs = [
        uav_label
        for uav_label in uav_labels
        if not shares_by_uav.get(uav_label)
    ]

    if missing_uavs:
        print(
            f"[WARN] Skipping {label}, UAVs={uav_count}; "
            f"missing data for {missing_uavs}"
        )
        return None

    data = [
        shares_by_uav[uav_label]
        for uav_label in uav_labels
    ]

    all_values = np.concatenate(
        [
            np.asarray(distribution, dtype=float)
            for distribution in data
            if distribution
        ]
    )

    if all_values.size == 0:
        print(
            f"[WARN] No share values for "
            f"{label}, UAVs={uav_count}"
        )
        return None

    # Same sizing as Analysis.py contribution boxplots.
    fig, ax = plt.subplots(
        figsize=(1.0 * len(uav_labels) + 3, 6)
    )

    bp = ax.boxplot(
        data,
        tick_labels=uav_labels,
        patch_artist=True,
    )

    # Same C0 blue boxes and black borders.
    for box in bp["boxes"]:
        box.set_facecolor("C0")
        box.set_edgecolor("black")

    # Same orange medians.
    for median in bp["medians"]:
        median.set(
            color="orange",
            linewidth=2,
        )

    ax.set_ylabel("Share of total revenue rate")

    if show_title:
        ax.set_title(label)

    # Same dynamic percentage y-axis logic as Analysis.py.
    max_share = float(all_values.max())

    if max_share <= 0.0:
        top_fraction = 0.1
    else:
        top_fraction = min(
            1.0,
            math.ceil(max_share * 10.0) / 10.0,
        )

    ax.set_ylim(0.0, top_fraction)

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    tick_step = 0.05 if top_fraction <= 0.5 else 0.1

    ax.set_yticks(
        np.arange(
            0.0,
            top_fraction + 1e-9,
            tick_step,
        )
    )

    plt.xticks(
        rotation=0,
        ha="center",
    )

    plt.tight_layout()

    safe_label = _safe_filename_label(label)

    output_path = (
        output_directory
        / f"UAVs{uav_count}_{safe_label}_per_uav_revenue_share.png"
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {output_path}")

    return output_path


def generate_per_uav_revenue_share_comparisons(
    sources: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
    uav_counts: Iterable[int] = range(3, 11),
    show_titles: bool = True,
) -> list[Path]:
    """
    Generate separate boxplots for each UAV count.

    For every |U|, the script creates plots for:
        1. Best Non-overlap configuration
        2. Best Overlap configuration
        3. Greedy
        4. Cluster+GA
        5. IRADA
    """
    set_plot_style()

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths: list[Path] = []

    for uav_count in uav_counts:
        uav_output_directory = (
            output_directory
            / f"UAVs{uav_count}"
        )

        uav_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"\n[INFO] Processing UAVs={uav_count}")

        for source_name, source in sources.items():
            result = _source_shares_for_uav_count(
                source_name=source_name,
                source=source,
                uav_count=uav_count,
            )

            if result is None:
                continue

            label, shares_by_uav = result

            output_path = _save_per_uav_share_boxplot(
                label=label,
                shares_by_uav=shares_by_uav,
                uav_count=uav_count,
                output_directory=uav_output_directory,
                show_title=show_titles,
            )

            if output_path is not None:
                output_paths.append(output_path)

    return output_paths


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    SOURCES = {
        # For every UAV count, choose the Non-overlap configuration
        # with the highest mean final total revenue rate.
        "Non-overlap": {
            "root": Path("results/non_overlap"),
            "workbook_pattern": "*.xlsx",
            "kind": "best_nonoverlap",
        },

        # For every UAV count, choose the Overlap configuration
        # with the highest mean final total revenue rate.
        "Overlap": {
            "root": Path("results/overlap"),
            "workbook_pattern": "*.xlsx",
            "kind": "best_overlap",
        },

        "Greedy": {
            "root": Path("results/greedy"),
            "workbook_pattern": "*Greedy*.xlsx",
            "kind": "single",
        },

        "Cluster+GA": {
            "root": Path("results/cluster_ga"),
            "workbook_pattern": "*.xlsx",
            "kind": "single",
        },

        "IRADA": {
            "root": Path("results/IRADA"),
            "workbook_pattern": "*IRADA*.xlsx",
            "kind": "single",
        },
    }

    generate_per_uav_revenue_share_comparisons(
        sources=SOURCES,
        output_directory=Path(
            "results/boxplots/per_uav_revenue_share_comparisons"
        ),
        uav_counts=range(3, 11),
        show_titles=True,
    )