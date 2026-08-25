from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _final_total_revenue_rates(workbook: Path) -> list[float]:
    """Return one final total revenue-rate value per workbook sheet/run."""
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

    final_rates: list[float] = []

    for sheet_name, frame in sheets.items():
        uav_columns = [
            column
            for column in frame.columns
            if str(column).upper().startswith("UAV")
        ]

        if frame.empty or not uav_columns:
            print(f"[WARN] Skipping {workbook.name}:{sheet_name}; no UAV data.")
            continue

        final_row = pd.to_numeric(
            frame.iloc[-1][uav_columns],
            errors="coerce",
        )

        final_total = float(final_row.sum())

        if np.isfinite(final_total):
            final_rates.append(final_total)

    return final_rates


def _matching_workbooks(
    root: Path,
    uav_count: int,
    workbook_pattern: str,
) -> list[Path]:
    """Find matching revenue workbooks below root for one UAV count."""
    if not root.exists():
        print(f"[WARN] Revenue root does not exist: {root}")
        return []

    matches: list[Path] = []

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

        matches.append(workbook)

    return sorted(matches)


def _nonoverlap_label(workbook: Path) -> str:
    """
    Convert a filename such as:

        UAVs3_GRID..._ModeGG_Random.xlsx
        UAVs3_GRID..._ModeGR_Sequential.xlsx

    into:

        NRGG
        NSGR
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

    return f"N{order_code}{mode_suffix}"


def _series_for_uav_count(
    sources: Mapping[str, Mapping[str, object]],
    uav_count: int,
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Build:

        {
            "NRGG": [...],
            "NSGG": [...],
            "IRADA": [...],
            "Greedy": [...],
            "Cluster+GA": [...],
        }

    for one UAV count.
    """
    series: dict[str, list[float]] = defaultdict(list)
    nonoverlap_labels: list[str] = []

    for source_name, source in sources.items():
        root = Path(source["root"])
        pattern = str(source.get("workbook_pattern", "*.xlsx"))
        source_kind = str(source.get("kind", "single")).lower()

        workbooks = _matching_workbooks(
            root=root,
            uav_count=uav_count,
            workbook_pattern=pattern,
        )

        if not workbooks:
            print(
                f"[WARN] No {source_name} workbooks "
                f"for UAVs={uav_count} in {root}"
            )
            continue

        for workbook in workbooks:
            rates = _final_total_revenue_rates(workbook)

            if not rates:
                continue

            if source_kind == "nonoverlap":
                label = _nonoverlap_label(workbook)

                if label not in nonoverlap_labels:
                    nonoverlap_labels.append(label)
            else:
                label = source_name

            series[label].extend(rates)

    # Same logic as Analysis.py:
    # NonOverlap labels first, then other algorithms
    nonoverlap_labels = sorted(nonoverlap_labels)

    ordered_labels = nonoverlap_labels.copy()

    benchmark_order = [
        "IRADA",
        "Greedy",
        "Cluster+GA",
    ]

    for label in benchmark_order:
        if label in series:
            ordered_labels.append(label)

    return series, ordered_labels


def plot_final_revenue_rate_comparison(
    sources: Mapping[str, Mapping[str, object]],
    uav_count: int,
    output_directory: str | Path,
    panel_label: str | None = None,
) -> Path | None:
    """
    Create one final-total revenue-rate boxplot for one UAV count.

    Non-overlap workbooks produce separate boxes:
        NRGG, NRGR, NRRG, NRRR, NSGG, NSGR, NSRG, NSRR.

    Greedy, Cluster+GA, and IRADA each produce one box.
    """
    series, labels = _series_for_uav_count(
        sources=sources,
        uav_count=uav_count,
    )

    labels = [
        label
        for label in labels
        if series[label]
    ]

    if not labels:
        print(f"[WARN] No comparison plot created for UAVs={uav_count}.")
        return None

    data = [series[label] for label in labels]

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / f"UAVs{uav_count}_final_total_revenue_rate.png"
    )

    fig, axis = plt.subplots(
        figsize=(1.2 * len(labels) + 4, 6)
    )

    boxplot = axis.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
    )

    for box in boxplot["boxes"]:
        box.set_facecolor("C0")
        box.set_edgecolor("black")

    for median in boxplot["medians"]:
        median.set(color="orange", linewidth=2)

    title_prefix = f"({panel_label}) " if panel_label else ""

    axis.set_title(f"{title_prefix}{uav_count} UAVs")
    axis.set_ylabel("Final total revenue rate")

    axis.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=90,
        ha="center",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {output_path}")

    return output_path


def generate_revenue_rate_comparison_images(
    sources: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
    uav_counts: Iterable[int] = range(3, 11),
) -> list[Path]:
    """Create one comparison image for every requested UAV count."""
    set_plot_style()

    output_paths: list[Path] = []

    for offset, uav_count in enumerate(uav_counts):
        panel_label = chr(ord("a") + offset)

        output_path = plot_final_revenue_rate_comparison(
            sources=sources,
            uav_count=uav_count,
            output_directory=output_directory,
            panel_label=panel_label,
        )

        if output_path is not None:
            output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    SOURCES = {
        # Each Non-overlap workbook creates a separate boxplot box:
        # NRGG, NRGR, NRRG, NRRR, NSGG, NSGR, NSRG, NSRR.
        "Non-overlapping": {
            "root": Path("results/non_overlap"),
            "workbook_pattern": "*.xlsx",
            "kind": "nonoverlap",
        },

        "IRADA": {
            "root": Path("results/IRADA"),
            "workbook_pattern": "*IRADA*.xlsx",
            "kind": "single",
        },

        "Greedy": {
            "root": Path("results/greedy"),
            "workbook_pattern": "*.xlsx",
            "kind": "single",
        },

        "Cluster+GA": {
            "root": Path("results/cluster_ga"),
            "workbook_pattern": "*.xlsx",
            "kind": "single",
        },


    }

    generate_revenue_rate_comparison_images(
        sources=SOURCES,
        output_directory=Path(
            "results/boxplots/revenue_rate_comparisons"
        ),
        uav_counts=range(3, 11),
    )