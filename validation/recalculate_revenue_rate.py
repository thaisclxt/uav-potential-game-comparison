from pathlib import Path
from typing import Dict, List

import pandas as pd

from algorithms.base_allocator import BaseAllocator
from src.config import load_configuration
from src.environment import GridEnvironment
from src.models import Waypoint
from src.utils import extract_num_uavs
from validation.algorithm_profiles import (
    ALGORITHM_PROFILES,
    AlgorithmProfile,
)

# Change this value to process another algorithm.
# Available examples:
# "IRADA", "greedy", "cluster_ga", "non_overlap", "overlap"
ALGORITHM_NAME = "IRADA"

GRID_SIZE = 13
MIN_UAVS = 3
MAX_UAVS = 10

PROFILE = ALGORITHM_PROFILES[ALGORITHM_NAME]

def load_waypoints(
    waypoint_file: Path,
    sheet_name: str,
) -> Dict[int, Waypoint]:
    """
    Load one waypoint worksheet as:

        {waypoint_id: Waypoint(...)}

    Required Excel columns:
        Waypoint, Revenue, X, Y
    """
    dataframe = pd.read_excel(
        waypoint_file,
        sheet_name=sheet_name,
    )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {
        "Waypoint",
        "Revenue",
        "X",
        "Y",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{waypoint_file.name} / {sheet_name} "
            f"is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return {
        int(row.Waypoint): Waypoint(
            wid=int(row.Waypoint),
            x=float(row.X),
            y=float(row.Y),
            revenue=float(row.Revenue),
        )
        for row in dataframe.itertuples(index=False)
    }


def parse_sequence(
    value: object,
    waypoint_by_id: Dict[int, Waypoint],
) -> List[Waypoint]:
    """
    Convert a saved sequence such as:

        "1-4-8"

    into:

        [
            waypoint_by_id[1],
            waypoint_by_id[4],
            waypoint_by_id[8],
        ]
    """
    if pd.isna(value):
        return []

    sequence_text = str(value).strip()

    if not sequence_text:
        return []

    waypoint_ids = [
        int(item.strip())
        for item in sequence_text.split("-")
        if item.strip()
    ]

    try:
        return [
            waypoint_by_id[waypoint_id]
            for waypoint_id in waypoint_ids
        ]
    except KeyError as exc:
        raise ValueError(
            f"Waypoint ID {exc.args[0]} is not present "
            "in the corresponding waypoint worksheet."
        ) from exc


def remove_duplicate_waypoints_in_sequence(
    sequence: List[Waypoint],
) -> List[Waypoint]:
    """
    Keep only the first occurrence of each waypoint ID
    within one UAV sequence.

    Example:
        [wp2, wp7, wp7] -> [wp2, wp7]

    This does not remove waypoints shared across different UAVs.
    """
    seen_waypoint_ids = set()
    cleaned_sequence: List[Waypoint] = []

    for waypoint in sequence:
        if waypoint.wid in seen_waypoint_ids:
            continue

        seen_waypoint_ids.add(waypoint.wid)
        cleaned_sequence.append(waypoint)

    return cleaned_sequence


def create_environment(
    waypoint_by_id: Dict[int, Waypoint],
    project_cfg,
    sim_cfg,
    grid_cfg,
    wp_cfg,
) -> GridEnvironment:
    """Create one environment for one simulation worksheet."""
    return GridEnvironment(
        project_configuration=project_cfg,
        simulation=sim_cfg,
        target_waypoints=list(waypoint_by_id.values()),
        width=grid_cfg.width,
        height=grid_cfg.height,
        spacing=grid_cfg.spacing,
        depot_location=grid_cfg.depot_location,
        wp_base_revenue=wp_cfg.base_revenue,
        wp_min_revenue=wp_cfg.min_revenue,
        wp_max_revenue=wp_cfg.max_revenue,
        number_targets=wp_cfg.number_targets,
        revenue_matrix=wp_cfg.revenue_matrix,
    )


def calculate_sheet_revenue_rates(
    tour_dataframe: pd.DataFrame,
    waypoint_file: Path,
    sheet_name: str,
    num_uavs: int,
    project_cfg,
    sim_cfg,
    grid_cfg,
    uav_cfg,
    wp_cfg,
    profile: AlgorithmProfile,
) -> pd.DataFrame:
    """
    Recalculate per-UAV revenue rates for every stored
    negotiation/optimization round in one worksheet.

    Output columns:
        <round column> | UAV0 | UAV1 | ... | UAV(n-1)
    """
    output_columns = [
        profile.round_column,
        *[
            f"UAV{uav_id}"
            for uav_id in range(num_uavs)
        ],
    ]

    if tour_dataframe.empty:
        return pd.DataFrame(columns=output_columns)

    if profile.round_column not in tour_dataframe.columns:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing the expected "
            f"round column '{profile.round_column}'."
        )

    waypoint_by_id = load_waypoints(
        waypoint_file=waypoint_file,
        sheet_name=sheet_name,
    )

    environment = create_environment(
        waypoint_by_id=waypoint_by_id,
        project_cfg=project_cfg,
        sim_cfg=sim_cfg,
        grid_cfg=grid_cfg,
        wp_cfg=wp_cfg,
    )

    allocator = BaseAllocator(
        environment=environment,
        num_uavs=num_uavs,
        uav_speed=uav_cfg.speed,
        max_flight_time=uav_cfg.max_flight_time,
    )

    output_rows: List[Dict[str, float | int]] = []

    for row in tour_dataframe.itertuples(index=False):
        row_data = row._asdict()

        revenue_row: Dict[str, float | int] = {
            profile.round_column: row_data[
                profile.round_column
            ],
        }

        for uav_id in range(num_uavs):
            sequence_column = (
                f"{profile.sequence_prefix}{uav_id}"
            )

            m_column = (
                f"{profile.m_prefix}{uav_id}"
            )

            sequence_value = row_data.get(sequence_column)
            m_j_value = row_data.get(m_column)

            if sequence_value is None:
                revenue_row[f"UAV{uav_id}"] = 0.0
                continue

            sequence = parse_sequence(
                value=sequence_value,
                waypoint_by_id=waypoint_by_id,
            )

            if profile.name == "overlap":
                sequence = remove_duplicate_waypoints_in_sequence(sequence)

            # stored_m_j = (
            #     0
            #     if pd.isna(m_j_value)
            #     else int(m_j_value)
            # )

            # if profile.name == "overlap":
            #     m_j = allocator._compute_m_j(sequence)
            # else:
            #     m_j = stored_m_j

            uav = allocator.uavs[uav_id]
            uav.sequence = sequence
            # uav.m_j = m_j

            # Determine m_j according to the algorithm profile.
            if profile.m_j_mode == "one":
                # IRADA: one depot-to-depot route.
                uav.m_j = 1 if uav.sequence else 0

            elif profile.m_j_mode == "recalculate":
                # Overlap: route may have changed after duplicate removal.
                uav.m_j = allocator._compute_m_j(
                    uav.sequence
                )

            else:
                # Greedy, Cluster+GA, and Non-Overlap:
                # use saved m_j from the sequence workbook.
                uav.m_j = (
                    0
                    if m_j_value is None or pd.isna(m_j_value)
                    else int(m_j_value)
                )

            revenue_row[f"UAV{uav_id}"] = (
                allocator.compute_revenue_rate(uav)
            )

        output_rows.append(revenue_row)

    return pd.DataFrame(
        output_rows,
        columns=output_columns,
    )


def calculate_excel_file(
    tour_file: Path,
    waypoint_file: Path,
    project_cfg,
    sim_cfg,
    grid_cfg,
    uav_cfg,
    wp_cfg,
    profile: AlgorithmProfile,
) -> Dict[str, pd.DataFrame]:
    """
    Recalculate revenue-rate DataFrames for all sheets
    in one tour Excel workbook.
    """
    num_uavs = extract_num_uavs(tour_file)

    print(
        f"\n[REVENUE] Processing {tour_file.name} "
        f"({num_uavs} UAVs)"
    )

    tour_xls = pd.ExcelFile(tour_file)

    revenue_sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name in tour_xls.sheet_names:
        tour_dataframe = pd.read_excel(
            tour_file,
            sheet_name=sheet_name,
        )

        tour_dataframe.columns = (
            tour_dataframe.columns
            .astype(str)
            .str.strip()
        )

        revenue_sheets[sheet_name] = (
            calculate_sheet_revenue_rates(
                tour_dataframe=tour_dataframe,
                waypoint_file=waypoint_file,
                sheet_name=sheet_name,
                num_uavs=num_uavs,
                project_cfg=project_cfg,
                sim_cfg=sim_cfg,
                grid_cfg=grid_cfg,
                uav_cfg=uav_cfg,
                wp_cfg=wp_cfg,
                profile=profile,
            )
        )

        print(
            f"[REVENUE] {sheet_name}: "
            f"{len(tour_dataframe)} round(s) processed."
        )

    return revenue_sheets


def save_revenue_file(
    tour_file: Path,
    revenue_sheets: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """
    Save standardized revenue-rate results.

    Example:
        Input:
            UAVs3_GRID13_1920_16_ModeGG_Random_sequences.xlsx

        Output:
            UAVs3_GRID13_ModeGG_Random_revenue_rate.xlsx
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_parts = tour_file.stem.split("_")

    # Expected input filename pattern:
    #
    # UAVs3_GRID13_1920_16_ModeGG_Random_sequences
    #
    # Keep:
    # UAVs3_GRID13_ModeGG_Random
    #
    # Remove:
    # 1920, 16, sequences

    if len(filename_parts) < 6:
        raise ValueError(
            f"Unexpected tour filename format: "
            f"{tour_file.name}"
        )

    output_stem = "_".join(
        [
            filename_parts[0],  # UAVs3
            filename_parts[1],  # GRID13
            *filename_parts[4:-1],  # ModeGG, Random
        ]
    )

    output_file = output_dir / (
        f"{output_stem}_revenue_rate.xlsx"
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:
        for sheet_name, dataframe in revenue_sheets.items():
            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    return output_file


def main() -> None:
    (
        project_cfg,
        sim_cfg,
        grid_cfg,
        uav_cfg,
        wp_cfg,
        _,
    ) = load_configuration(
        Path("settings.yaml")
    )

    print(
        f"\n[REVENUE] Selected algorithm profile: "
        f"{PROFILE.name}"
    )

    for num_uavs in range(
        MIN_UAVS,
        MAX_UAVS + 1,
    ):
        scenario_name = (
            f"UAVs{num_uavs}_GRID{GRID_SIZE}"
        )

        tour_dir = (
            PROFILE.results_dir
            / scenario_name
            / PROFILE.tour_dir_name
        )

        waypoint_file = (
            PROFILE.waypoints_dir
            / f"{scenario_name}_waypoints.xlsx"
        )

        output_dir = (
            PROFILE.results_dir
            / scenario_name
            / PROFILE.output_dir_name
        )

        if not tour_dir.exists():
            print(
                f"\n[REVENUE] Skipping {scenario_name}: "
                "tour directory not found:\n"
                f"  {tour_dir.resolve()}"
            )
            continue

        if not waypoint_file.exists():
            print(
                f"\n[REVENUE] Skipping {scenario_name}: "
                "waypoint file not found:\n"
                f"  {waypoint_file.resolve()}"
            )
            continue

        tour_files = sorted(
            tour_dir.glob("*.xlsx")
        )

        if not tour_files:
            print(
                f"\n[REVENUE] Skipping {scenario_name}: "
                "no Excel tour files found."
            )
            continue

        print(
            f"\n{'=' * 60}\n"
            f"[REVENUE] Algorithm: {PROFILE.name}\n"
            f"[REVENUE] Scenario: {scenario_name}\n"
            f"[REVENUE] Tour files: {len(tour_files)}\n"
            f"{'=' * 60}"
        )

        for tour_file in tour_files:
            try:
                revenue_sheets = calculate_excel_file(
                    tour_file=tour_file,
                    waypoint_file=waypoint_file,
                    project_cfg=project_cfg,
                    sim_cfg=sim_cfg,
                    grid_cfg=grid_cfg,
                    uav_cfg=uav_cfg,
                    wp_cfg=wp_cfg,
                    profile=PROFILE,
                )

                output_file = save_revenue_file(
                    tour_file=tour_file,
                    revenue_sheets=revenue_sheets,
                    output_dir=output_dir,
                )

                print(
                    f"[REVENUE] Saved:\n"
                    f"  {output_file.resolve()}"
                )

            except Exception as exc:
                print(
                    f"[REVENUE] Failed: {tour_file.name}\n"
                    f"Reason: {exc}"
                )


if __name__ == "__main__":
    main()