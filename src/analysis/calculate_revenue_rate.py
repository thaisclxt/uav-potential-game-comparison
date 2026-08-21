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

RESULTS_DIR = Path("results/non_overlap")
WAYPOINTS_DIR = Path("waypoints")

GRID_SIZE = 13
MIN_UAVS = 3
MAX_UAVS = 10


# ============================================================
# Helpers
# ============================================================

def extract_num_uavs(excel_file: Path) -> int:
    """Extract UAV count from a filename such as UAVs3_GRID13_...xlsx."""
    try:
        return int(
            excel_file.stem.split("_")[0].replace("UAVs", "")
        )
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine UAV count from {excel_file.name}."
        ) from exc


def load_waypoints(
    waypoint_file: Path,
    sheet_name: str,
) -> Dict[int, Waypoint]:
    """Load one waypoint worksheet as {waypoint_id: Waypoint}."""
    dataframe = pd.read_excel(
        waypoint_file,
        sheet_name=sheet_name,
    )

    dataframe.columns = dataframe.columns.astype(str).str.strip()

    required_columns = {"Waypoint", "Revenue", "X", "Y"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{waypoint_file.name} / {sheet_name} is missing "
            f"columns: {sorted(missing_columns)}"
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
    """Convert '1-4-8' into a list of Waypoint objects."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    waypoint_ids = (
        int(item.strip())
        for item in text.split("-")
        if item.strip()
    )

    try:
        return [
            waypoint_by_id[waypoint_id]
            for waypoint_id in waypoint_ids
        ]
    except KeyError as exc:
        raise ValueError(
            f"Waypoint ID {exc.args[0]} is not present "
            "in the corresponding waypoint sheet."
        ) from exc


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


# ============================================================
# Revenue-rate calculation
# ============================================================

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
) -> pd.DataFrame:
    """
    Calculate all UAV revenue rates for every negotiation round
    in one tour worksheet.
    """
    if tour_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "negotiation_round",
                *[
                    f"UAV{uav_id}"
                    for uav_id in range(num_uavs)
                ],
            ]
        )

    if "negotiation_round" not in tour_dataframe.columns:
        raise ValueError(
            f"Sheet '{sheet_name}' has no negotiation_round column."
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
            "negotiation_round": row_data["negotiation_round"],
        }

        for uav_id in range(num_uavs):
            sequence_column = f"UAV{uav_id}"
            m_column = f"m_{uav_id}"

            sequence_value = row_data.get(sequence_column)
            m_j_value = row_data.get(m_column)

            if sequence_value is None or m_j_value is None:
                revenue_row[f"UAV{uav_id}"] = 0.0
                continue

            sequence = parse_sequence(
                value=sequence_value,
                waypoint_by_id=waypoint_by_id,
            )

            m_j = 0 if pd.isna(m_j_value) else int(m_j_value)

            uav = allocator.uavs[uav_id]
            uav.sequence = sequence
            uav.m_j = m_j

            revenue_row[f"UAV{uav_id}"] = (
                allocator.compute_revenue_rate(uav)
            )

        output_rows.append(revenue_row)

    return pd.DataFrame(output_rows)


# ============================================================
# Process one Excel workbook
# ============================================================

def calculate_excel_file(
    tour_file: Path,
    waypoint_file: Path,
    project_cfg,
    sim_cfg,
    grid_cfg,
    uav_cfg,
    wp_cfg,
) -> Dict[str, pd.DataFrame]:
    """
    Create a revenue-rate DataFrame for every sheet
    in a tour workbook.
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
            tour_dataframe.columns.astype(str).str.strip()
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
            )
        )

        print(
            f"[REVENUE] {sheet_name}: "
            f"{len(tour_dataframe)} negotiation rounds processed."
        )

    return revenue_sheets


# ============================================================
# Save result workbook
# ============================================================

def save_revenue_file(
    tour_file: Path,
    revenue_sheets: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Write all revenue-rate sheets to one Excel workbook."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"{tour_file.stem}_revenue_rates.xlsx"
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    (
        project_cfg,
        sim_cfg,
        grid_cfg,
        uav_cfg,
        wp_cfg,
    ) = load_configuration(
        Path("settings.yaml")
    )

    for num_uavs in range(
        MIN_UAVS,
        MAX_UAVS + 1,
    ):
        scenario_name = (
            f"UAVs{num_uavs}_GRID{GRID_SIZE}"
        )

        tour_dir = (
            RESULTS_DIR
            / scenario_name
            / "tour"
        )

        waypoint_file = (
            WAYPOINTS_DIR
            / f"{scenario_name}_waypoints.xlsx"
        )

        output_dir = (
            RESULTS_DIR
            / scenario_name
            / "new_revenue"
        )

        if not tour_dir.exists():
            print(
                f"\n[REVENUE] Skipping {scenario_name}: "
                f"tour directory not found:\n"
                f"  {tour_dir.resolve()}"
            )
            continue

        if not waypoint_file.exists():
            print(
                f"\n[REVENUE] Skipping {scenario_name}: "
                f"waypoint file not found:\n"
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
                )

                output_file = save_revenue_file(
                    tour_file=tour_file,
                    revenue_sheets=revenue_sheets,
                    output_dir=output_dir,
                )

                print(
                    f"[REVENUE] Saved: "
                    f"{output_file.resolve()}"
                )

            except Exception as exc:
                print(
                    f"[REVENUE] Failed: {tour_file.name}\n"
                    f"Reason: {exc}"
                )


if __name__ == "__main__":
    main()