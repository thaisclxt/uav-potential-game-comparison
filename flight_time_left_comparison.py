from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# Configuration
# ============================================================

UAV_SPEED = 16.0
MAX_FLIGHT_TIME = 1920.0
MAX_SHEETS = 100


# ============================================================
# Geometry and labels
# ============================================================

def euclidean_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def _uav_sort_key(column: object) -> tuple[int, str]:
    text = str(column)
    suffix = text.upper().replace("UAV", "")

    try:
        return int(suffix), text
    except ValueError:
        return 999999, text


def _mode_label(
    sequence_workbook: Path,
    prefix: str,
) -> str:
    """
    Examples:
        ..._ModeGG_Random_sequences.xlsx
            -> NRGG / ORGG

        ..._ModeGR_Sequential_sequences.xlsx
            -> NSGR / OSGR
    """
    stem = sequence_workbook.stem.replace("_sequences", "")
    parts = stem.split("_")

    mode_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.startswith("Mode")
        ),
        None,
    )

    if mode_index is None:
        return sequence_workbook.stem

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
# Workbook discovery
# ============================================================

def _matching_sequence_workbooks(
    sequence_root: Path,
    uav_count: int,
    workbook_pattern: str,
) -> list[Path]:
    """
    Find sequence workbooks whose filename starts with UAVsN_.

    Example:
        UAVs3_GRID13_..._sequences.xlsx
    """
    if not sequence_root.exists():
        print(f"[WARN] Sequence root does not exist: {sequence_root}")
        return []

    workbooks: list[Path] = []

    for workbook in sequence_root.rglob(workbook_pattern):
        if not workbook.is_file():
            continue

        if not workbook.name.startswith(f"UAVs{uav_count}_"):
            continue

        if "_sequences" not in workbook.stem:
            continue

        workbooks.append(workbook)

    return sorted(workbooks)


def _find_waypoint_workbook(
    sequence_workbook: Path,
    waypoint_root: Path,
    uav_count: int,
) -> Path | None:
    """
    First search near the sequence workbook:

        .../UAVsN_GRID.../sequences/*.xlsx
        .../UAVsN_GRID.../waypoints/UAVsN_GRID..._waypoints.xlsx

    Then search below waypoint_root.
    """
    local_waypoint_dir = sequence_workbook.parent.parent / "waypoints"

    local_candidates = sorted(
        local_waypoint_dir.glob(
            f"UAVs{uav_count}_GRID*_waypoints.xlsx"
        )
    )

    if local_candidates:
        return local_candidates[0]

    fallback_candidates = sorted(
        waypoint_root.rglob(
            f"UAVs{uav_count}_GRID*_waypoints.xlsx"
        )
    )

    if fallback_candidates:
        return fallback_candidates[0]

    print(
        f"[WARN] No waypoint workbook found for "
        f"{sequence_workbook.name}"
    )

    return None


# ============================================================
# Waypoint cache
# ============================================================

def _load_waypoint_coordinate_cache(
    waypoint_workbook: Path,
) -> dict[str, dict[int, tuple[float, float]]]:
    """
    Read the first MAX_SHEETS waypoint sheets once.

    Return structure:

        {
            "SimRun0": {
                0: (0.0, 60.0),
                1: (0.0, 120.0),
            },
            "SimRun1": {...},
        }

    No index_col is used here because Waypoint must remain a column.
    """
    try:
        with pd.ExcelFile(waypoint_workbook) as excel_file:
            sheet_names = excel_file.sheet_names[:MAX_SHEETS]

            waypoint_sheets = pd.read_excel(
                excel_file,
                sheet_name=sheet_names,
                usecols=["Waypoint", "X", "Y"],
                na_filter=False,
            )
    except Exception as exc:
        print(f"[WARN] Could not read {waypoint_workbook}: {exc}")
        return {}

    coordinate_cache: dict[str, dict[int, tuple[float, float]]] = {}

    for sheet_name, frame in waypoint_sheets.items():
        required_columns = {"Waypoint", "X", "Y"}

        if not required_columns.issubset(frame.columns):
            print(
                f"[WARN] Waypoint sheet {sheet_name} in "
                f"{waypoint_workbook.name} is missing Waypoint, X, or Y."
            )
            continue

        coordinate_cache[sheet_name] = {
            int(row.Waypoint): (
                float(row.X),
                float(row.Y),
            )
            for _, row in frame.iterrows()
        }

    return coordinate_cache


def _coordinates_for_run(
    coordinate_cache: dict[str, dict[int, tuple[float, float]]],
    run_name: str,
) -> dict[int, tuple[float, float]]:
    """Return coordinates for run_name or fall back to the first sheet."""
    if run_name in coordinate_cache:
        return coordinate_cache[run_name]

    if coordinate_cache:
        return next(iter(coordinate_cache.values()))

    return {}


# ============================================================
# Sequence and tour-time calculation
# ============================================================

def _parse_sequence(sequence_value: object) -> list[int]:
    """Convert '0-2-5' into [0, 2, 5]."""
    text = str(sequence_value)

    if not text or text.lower() == "nan":
        return []

    return [
        int(value)
        for value in text.split("-")
        if value and value.lower() != "nan"
    ]


def _full_repeated_tour_time(
    waypoint_ids: list[int],
    coordinates: dict[int, tuple[float, float]],
    m_j: int,
    uav_speed: float,
) -> float:
    """
    Depot -> first
    + m_j * internal route
    + (m_j - 1) * last-to-first closure
    + last -> depot
    """
    if not waypoint_ids or m_j < 1:
        return 0.0

    depot = (0.0, 0.0)

    points = [
        coordinates[waypoint_id]
        for waypoint_id in waypoint_ids
        if waypoint_id in coordinates
    ]

    if len(points) != len(waypoint_ids):
        return 0.0

    outbound_distance = euclidean_distance(depot, points[0])
    return_distance = euclidean_distance(points[-1], depot)

    internal_distance = sum(
        euclidean_distance(point_a, point_b)
        for point_a, point_b in zip(points[:-1], points[1:])
    )

    closure_distance = (
        euclidean_distance(points[-1], points[0])
        if len(points) > 1
        else 0.0
    )

    total_distance = (
        outbound_distance
        + m_j * internal_distance
        + (m_j - 1) * closure_distance
        + return_distance
    )

    return total_distance / float(uav_speed)


def _irada_tour_time(
    waypoint_ids: list[int],
    coordinates: dict[int, tuple[float, float]],
    uav_speed: float,
) -> float:
    """IRADA uses one complete depot-to-depot route."""
    if not waypoint_ids:
        return 0.0

    depot = (0.0, 0.0)

    points = [
        coordinates[waypoint_id]
        for waypoint_id in waypoint_ids
        if waypoint_id in coordinates
    ]

    if len(points) != len(waypoint_ids):
        return 0.0

    total_distance = euclidean_distance(depot, points[0])

    total_distance += sum(
        euclidean_distance(point_a, point_b)
        for point_a, point_b in zip(points[:-1], points[1:])
    )

    total_distance += euclidean_distance(points[-1], depot)

    return total_distance / float(uav_speed)


def _flight_time_left_for_sequence_workbook(
    sequence_workbook: Path,
    coordinate_cache: dict[str, dict[int, tuple[float, float]]],
    algorithm_kind: str,
    uav_speed: float,
    max_flight_time: float,
) -> list[float]:
    """
    Calculate flight-time-left values from the final round of the first
    MAX_SHEETS sequence workbook sheets.
    """
    try:
        with pd.ExcelFile(sequence_workbook) as excel_file:
            sheet_names = excel_file.sheet_names[:MAX_SHEETS]

            sequence_sheets = pd.read_excel(
                excel_file,
                sheet_name=sheet_names,
                index_col=0,
            )
    except Exception as exc:
        print(f"[WARN] Could not read {sequence_workbook}: {exc}")
        return []

    values: list[float] = []

    for run_name, sequence_frame in sequence_sheets.items():
        coordinates = _coordinates_for_run(
            coordinate_cache=coordinate_cache,
            run_name=run_name,
        )

        if not coordinates:
            continue

        uav_columns = sorted(
            [
                column
                for column in sequence_frame.columns
                if str(column).upper().startswith("UAV")
            ],
            key=_uav_sort_key,
        )

        if sequence_frame.empty or not uav_columns:
            continue

        last_row = sequence_frame.iloc[-1]

        for uav_index, uav_column in enumerate(uav_columns):
            waypoint_ids = _parse_sequence(last_row[uav_column])

            if not waypoint_ids:
                continue

            # Same behavior as Analysis.py:
            # A one-waypoint sequence is treated as a hovering/depletion case.
            if len(waypoint_ids) == 1:
                values.append(0.0)
                continue

            if algorithm_kind == "irada":
                total_time = _irada_tour_time(
                    waypoint_ids=waypoint_ids,
                    coordinates=coordinates,
                    uav_speed=uav_speed,
                )
            else:
                m_column = f"m_{uav_index}"

                if m_column not in sequence_frame.columns:
                    print(
                        f"[WARN] Missing {m_column} in "
                        f"{sequence_workbook.name}; "
                        f"skipping {uav_column}."
                    )
                    continue

                try:
                    m_j = int(last_row[m_column])
                except (TypeError, ValueError):
                    continue

                if m_j < 1:
                    continue

                total_time = _full_repeated_tour_time(
                    waypoint_ids=waypoint_ids,
                    coordinates=coordinates,
                    m_j=m_j,
                    uav_speed=uav_speed,
                )

            flight_time_left = max(
                float(max_flight_time) - total_time,
                0.0,
            )

            values.append(flight_time_left)

    return values


# ============================================================
# Data collection
# ============================================================

def _label_for_source_workbook(
    source_name: str,
    source: Mapping[str, object],
    sequence_workbook: Path,
) -> str:
    source_kind = str(source.get("kind", "single")).lower()

    if source_kind == "nonoverlap":
        return _mode_label(
            sequence_workbook=sequence_workbook,
            prefix="N",
        )

    if source_kind == "overlap":
        return _mode_label(
            sequence_workbook=sequence_workbook,
            prefix="O",
        )

    return source_name


def _collect_flight_time_left(
    sources: Mapping[str, Mapping[str, object]],
    uav_count: int,
    uav_speed: float,
    max_flight_time: float,
) -> dict[str, list[float]]:
    """
    Read every algorithm's final sequences and collect remaining flight time.

    The same waypoint workbook is loaded only once and reused.
    """
    label_to_values: dict[str, list[float]] = defaultdict(list)

    waypoint_coordinate_cache: dict[
        Path,
        dict[str, dict[int, tuple[float, float]]]
    ] = {}

    for source_name, source in sources.items():
        sequence_root = Path(source["sequence_root"])

        waypoint_root = Path(
            source.get("waypoint_root", sequence_root)
        )

        sequence_pattern = str(
            source.get(
                "sequence_pattern",
                "*_sequences.xlsx",
            )
        )

        source_kind = str(source.get("kind", "single")).lower()

        sequence_workbooks = _matching_sequence_workbooks(
            sequence_root=sequence_root,
            uav_count=uav_count,
            workbook_pattern=sequence_pattern,
        )

        if not sequence_workbooks:
            print(
                f"[WARN] No sequence workbooks for "
                f"{source_name}, UAVs={uav_count}"
            )
            continue

        for sequence_workbook in sequence_workbooks:
            waypoint_workbook = _find_waypoint_workbook(
                sequence_workbook=sequence_workbook,
                waypoint_root=waypoint_root,
                uav_count=uav_count,
            )

            if waypoint_workbook is None:
                continue

            waypoint_key = waypoint_workbook.resolve()

            if waypoint_key not in waypoint_coordinate_cache:
                print(
                    f"[INFO] Loading waypoint workbook once: "
                    f"{waypoint_workbook.name}"
                )

                waypoint_coordinate_cache[waypoint_key] = (
                    _load_waypoint_coordinate_cache(
                        waypoint_workbook=waypoint_workbook
                    )
                )

            coordinate_cache = waypoint_coordinate_cache[waypoint_key]

            if not coordinate_cache:
                continue

            label = _label_for_source_workbook(
                source_name=source_name,
                source=source,
                sequence_workbook=sequence_workbook,
            )

            values = _flight_time_left_for_sequence_workbook(
                sequence_workbook=sequence_workbook,
                coordinate_cache=coordinate_cache,
                algorithm_kind=source_kind,
                uav_speed=uav_speed,
                max_flight_time=max_flight_time,
            )

            label_to_values[label].extend(values)

    return label_to_values


# ============================================================
# Plotting
# ============================================================

def _ordered_labels(
    label_to_values: dict[str, list[float]],
) -> list[str]:
    """
    Required order:

    NRGG ... NSRR
    ORGG ... OSRR
    IRADA
    Greedy
    Cluster+GA
    """
    nonoverlap_labels = sorted(
        label
        for label in label_to_values
        if label.startswith("N")
    )

    overlap_labels = sorted(
        label
        for label in label_to_values
        if label.startswith("O")
    )

    labels = nonoverlap_labels + overlap_labels

    for label in ["IRADA", "Greedy", "Cluster+GA"]:
        if label in label_to_values:
            labels.append(label)

    return labels


def plot_flight_time_left_for_uav_count(
    sources: Mapping[str, Mapping[str, object]],
    uav_count: int,
    output_directory: str | Path,
    uav_speed: float,
    max_flight_time: float,
) -> Path | None:
    """Create one combined boxplot for one UAV count."""
    label_to_values = _collect_flight_time_left(
        sources=sources,
        uav_count=uav_count,
        uav_speed=uav_speed,
        max_flight_time=max_flight_time,
    )

    labels = _ordered_labels(label_to_values)

    labels = [
        label
        for label in labels
        if label_to_values[label]
    ]

    if not labels:
        print(
            f"[WARN] No flight-time-left data "
            f"for UAVs={uav_count}."
        )
        return None

    data = [
        label_to_values[label]
        for label in labels
    ]

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / f"{uav_count}uavs.png"

    # Same style and dynamic width as Analysis.py.
    fig_width = 4.0 + 0.25 * len(labels)

    fig, ax = plt.subplots(
        figsize=(fig_width, 4.5),
        layout="constrained",
    )

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
    )

    for box in bp["boxes"]:
        box.set_facecolor("C0")
        box.set_edgecolor("black")

    for median in bp["medians"]:
        median.set(
            color="orange",
            linewidth=2,
        )

    ax.set_ylabel("Flight time left (s)")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {output_path}")

    return output_path


def generate_flight_time_left_boxplots(
    sources: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
    uav_speed: float,
    max_flight_time: float,
    uav_counts: Iterable[int] = range(3, 11),
) -> list[Path]:
    """Generate results/flight_time_left/3uavs.png through 10uavs.png."""
    set_plot_style()

    output_paths: list[Path] = []

    for uav_count in uav_counts:
        print(f"\n[INFO] Processing UAVs={uav_count}")

        output_path = plot_flight_time_left_for_uav_count(
            sources=sources,
            uav_count=uav_count,
            output_directory=output_directory,
            uav_speed=uav_speed,
            max_flight_time=max_flight_time,
        )

        if output_path is not None:
            output_paths.append(output_path)

    return output_paths


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    SOURCES = {
        "Non-overlap": {
            "sequence_root": Path("results/non_overlap"),
            "waypoint_root": Path("waypoints"),
            "sequence_pattern": "*.xlsx",
            "kind": "nonoverlap",
        },

        "Overlap": {
            "sequence_root": Path("results/overlap"),
            "waypoint_root": Path("overlap_waypoints"),
            "sequence_pattern": "*.xlsx",
            "kind": "overlap",
        },

        "IRADA": {
            "sequence_root": Path("results/IRADA"),
            "waypoint_root": Path("waypoints"),
            "sequence_pattern": "*.xlsx",
            "kind": "irada",
        },

        "Greedy": {
            "sequence_root": Path("results/greedy"),
            "waypoint_root": Path("waypoints"),
            "sequence_pattern": "*.xlsx",
            "kind": "single",
        },

        "Cluster+GA": {
            "sequence_root": Path("results/cluster_ga"),
            "waypoint_root": Path("waypoints"),
            "sequence_pattern": "*.xlsx",
            "kind": "single",
        },
    }

    generate_flight_time_left_boxplots(
        sources=SOURCES,
        output_directory=Path("results/boxplots/flight_time_left"),
        uav_speed=UAV_SPEED,
        max_flight_time=MAX_FLIGHT_TIME,
        uav_counts=range(3, 11),
    )