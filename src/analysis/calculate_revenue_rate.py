from pathlib import Path
from typing import Dict, List

import pandas as pd

from algorithms.base_allocator import BaseAllocator
from src.config import load_configuration
from src.environment import GridEnvironment
from src.models import Waypoint


# ============================================================
# Paths
# ============================================================

TOUR_DIR = Path(
    "results/non_overlap/UAVs3_GRID13/tour"
)

WAYPOINT_FILE = Path(
    "waypoints/UAVs3_GRID13_waypoints.xlsx"
)

OUTPUT_DIR = Path(
    "results/non_overlap/UAVs3_GRID13/new_revenue"
)

def load_waypoints(
    waypoint_file: Path,
    sheet_name: str,
) -> Dict[int, Waypoint]:
    """
    Load waypoints from one worksheet.

    Expected waypoint columns:
        Waypoint, Revenue, X, Y
    """
    dataframe = pd.read_excel(
        waypoint_file,
        sheet_name=sheet_name,
    )

    dataframe.columns = dataframe.columns.str.strip()

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
            f"Missing columns in {waypoint_file.name}, "
            f"sheet '{sheet_name}': {missing_columns}"
        )

    waypoints: Dict[int, Waypoint] = {}

    for _, row in dataframe.iterrows():
        waypoint = Waypoint(
            wid=int(row["Waypoint"]),
            x=float(row["X"]),
            y=float(row["Y"]),
            revenue=float(row["Revenue"]),
        )

        waypoints[waypoint.wid] = waypoint

    return waypoints


def parse_sequence(
    sequence_value: object,
    waypoints: Dict[int, Waypoint],
) -> List[Waypoint]:
    """
    Convert a stored tour sequence such as:

        "1-5-9-12"

    into a list of matching Waypoint objects.
    """
    if pd.isna(sequence_value):
        return []

    sequence_text = str(sequence_value).strip()

    if not sequence_text:
        return []

    waypoint_ids = [
        int(value.strip())
        for value in sequence_text.split("-")
        if value.strip()
    ]

    sequence: List[Waypoint] = []

    for waypoint_id in waypoint_ids:
        if waypoint_id not in waypoints:
            raise ValueError(
                f"Waypoint ID {waypoint_id} was not found "
                "in the corresponding waypoint worksheet."
            )

        sequence.append(
            waypoints[waypoint_id]
        )

    return sequence


def extract_num_uavs(
    excel_file: Path,
) -> int:
    """
    Example filename:

        UAVs3_GRID13_600_10_cluster_ga_sequences.xlsx

    Returns:
        3
    """
    try:
        uav_part = excel_file.stem.split("_")[0]

        return int(
            uav_part.replace("UAVs", "")
        )

    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Could not determine the number of UAVs "
            f"from filename: {excel_file.name}"
        ) from exc


def validate_tour_columns(
    dataframe: pd.DataFrame,
    num_uavs: int,
    sheet_name: str,
) -> None:
    """
    Require negotiation_round and a sequence/m_j pair for
    every configured UAV.
    """
    required_columns = {
        "negotiation_round",
    }

    for uav_id in range(num_uavs):
        required_columns.add(f"UAV{uav_id}")
        required_columns.add(f"m_{uav_id}")

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns in sheet "
            f"'{sheet_name}': {sorted(missing_columns)}"
        )
    

def calculate_excel_file(
    excel_file: Path,
    waypoint_file: Path,
    project_cfg,
    sim_cfg,
    grid_cfg,
    uav_cfg,
    wp_cfg,
) -> Dict[str, pd.DataFrame]:
    """
    Read every worksheet and every negotiation-round row from
    an input tour workbook.

    Returns:
        {
            "SimRun1": dataframe_of_revenue_rates,
            "SimRun2": dataframe_of_revenue_rates,
            ...
        }

    Output DataFrame columns:
        negotiation_round, UAV0, UAV1, ..., UAV(n-1)
    """
    print(f"\n[REVENUE] Processing: {excel_file.name}")

    num_uavs = extract_num_uavs(excel_file)

    print(f"[REVENUE] Number of UAVs: {num_uavs}")

    tour_xls = pd.ExcelFile(excel_file)

    revenue_sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name in tour_xls.sheet_names:
        print(f"\n[REVENUE] Processing sheet: {sheet_name}")

        tour_dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
        )

        output_rows: List[Dict[str, float | int]] = []

        # Create one zero-rate output row for empty worksheets.
        if tour_dataframe.empty:
            print(
                "[REVENUE] Sheet is empty. "
                "Writing one row with zero revenue rates."
            )

            empty_row: Dict[str, float | int] = {
                "negotiation_round": 0,
            }

            for uav_id in range(num_uavs):
                empty_row[f"UAV{uav_id}"] = 0.0

            revenue_sheets[sheet_name] = pd.DataFrame(
                [empty_row]
            )

            continue

        tour_dataframe.columns = (
            tour_dataframe.columns
            .astype(str)
            .str.strip()
        )

        validate_tour_columns(
            dataframe=tour_dataframe,
            num_uavs=num_uavs,
            sheet_name=sheet_name,
        )

        # Load waypoint positions/revenues once for this SimRun sheet.
        waypoints = load_waypoints(
            waypoint_file=waypoint_file,
            sheet_name=sheet_name,
        )

        environment = GridEnvironment(
            project_configuration=project_cfg,
            simulation=sim_cfg,
            target_waypoints=list(waypoints.values()),
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

        # BaseAllocator provides shared repeated-tour and
        # revenue-rate calculations. It does not solve/optimize.
        allocator = BaseAllocator(
            environment=environment,
            num_uavs=num_uavs,
            uav_speed=uav_cfg.speed,
            max_flight_time=uav_cfg.max_flight_time,
        )

        for _, row in tour_dataframe.iterrows():
            negotiation_round = row["negotiation_round"]

            revenue_row: Dict[str, float | int] = {
                "negotiation_round": negotiation_round,
            }

            allocator.reset()

            for uav_id in range(num_uavs):
                sequence_column = f"UAV{uav_id}"
                m_column = f"m_{uav_id}"

                sequence = parse_sequence(
                    sequence_value=row[sequence_column],
                    waypoints=waypoints,
                )

                stored_m_j = row[m_column]

                m_j = (
                    0
                    if pd.isna(stored_m_j)
                    else int(stored_m_j)
                )

                uav = allocator.uavs[uav_id]
                uav.sequence = sequence
                uav.m_j = m_j

                revenue_rate = allocator.compute_revenue_rate(
                    uav
                )

                revenue_row[f"UAV{uav_id}"] = revenue_rate

            output_rows.append(revenue_row)

        revenue_sheets[sheet_name] = pd.DataFrame(
            output_rows
        )

        print(
            f"[REVENUE] Completed {len(output_rows)} "
            "negotiation round(s)."
        )

    return revenue_sheets


def save_revenue_file(
    input_file: Path,
    revenue_sheets: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """
    Save calculated revenue rates into a new workbook.

    The output workbook preserves the source worksheet names.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / (
        f"{input_file.stem}_revenue_rates.xlsx"
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
    settings_path = Path("settings.yaml")

    (
        project_cfg,
        sim_cfg,
        grid_cfg,
        uav_cfg,
        wp_cfg,
    ) = load_configuration(settings_path)

    tour_files = sorted(
        TOUR_DIR.glob("*.xlsx")
    )

    if not tour_files:
        print(
            f"[REVENUE] No .xlsx files found in:\n"
            f"  {TOUR_DIR.resolve()}"
        )
        return

    print(
        f"[REVENUE] Found {len(tour_files)} "
        "tour workbook(s)."
    )

    for tour_file in tour_files:
        try:
            revenue_sheets = calculate_excel_file(
                excel_file=tour_file,
                waypoint_file=WAYPOINT_FILE,
                project_cfg=project_cfg,
                sim_cfg=sim_cfg,
                grid_cfg=grid_cfg,
                uav_cfg=uav_cfg,
                wp_cfg=wp_cfg,
            )

            output_file = save_revenue_file(
                input_file=tour_file,
                revenue_sheets=revenue_sheets,
                output_dir=OUTPUT_DIR,
            )

            print(
                f"\n[REVENUE] Saved output workbook:\n"
                f"  {output_file.resolve()}"
            )

        except Exception as exc:
            print(
                f"\n[REVENUE] Failed to process "
                f"{tour_file.name}:\n"
                f"  {exc}"
            )


if __name__ == "__main__":
    main()