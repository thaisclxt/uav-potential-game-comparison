from pathlib import Path
from typing import Dict, List

import pandas as pd


RESULTS_DIR = Path("results/cluster_ga")
WAYPOINTS_DIR = Path("data/non_overlap_waypoints")

GRID_SIZE = 13
MIN_UAVS = 3
MAX_UAVS = 10


def extract_num_uavs(excel_file: Path) -> int:
    """Extract UAV count from filename such as UAVs3_GRID13_...xlsx."""
    try:
        return int(
            excel_file.stem.split("_")[0].replace("UAVs", "")
        )
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine UAV count from {excel_file.name}."
        ) from exc


def load_waypoint_ids(
    waypoint_file: Path,
    sheet_name: str,
) -> List[int]:
    """
    Load only the Waypoint IDs from one waypoint sheet.
    Expected columns: Waypoint, Revenue, X, Y (only Waypoint is used).
    """
    dataframe = pd.read_excel(
        waypoint_file,
        sheet_name=sheet_name,
    )

    dataframe.columns = dataframe.columns.astype(str).str.strip()

    if "Waypoint" not in dataframe.columns:
        raise ValueError(
            f"{waypoint_file.name} / {sheet_name} "
            "is missing a 'Waypoint' column."
        )

    return [
        int(wp)
        for wp in dataframe["Waypoint"]
    ]


def parse_waypoint_ids_from_sequence(
    sequence_value: object,
) -> List[int]:
    """
    Convert a sequence string such as '1-5-9-12'
    into [1, 5, 9, 12].
    """
    if pd.isna(sequence_value):
        return []

    text = str(sequence_value).strip()

    if not text:
        return []

    return [
        int(item.strip())
        for item in text.split("-")
        if item.strip()
    ]


def build_waypoint_uav_assignment_for_sheet(
    tour_dataframe: pd.DataFrame,
    waypoint_ids: List[int],
    num_uavs: int,
    sheet_name: str,
) -> pd.DataFrame:
    """
    For one simulation sheet, build a DataFrame:

        Waypoint | UAV

    where each row indicates that a given waypoint ID
    belongs to UAV k in that simulation.
    """
    if tour_dataframe.empty:
        # No routes; return empty assignment table
        return pd.DataFrame(
            columns=["Waypoint id", "UAV/Cluster/K"]
        )

    # Use the first row as the representative route for this SimRun
    row = tour_dataframe.iloc[0]

    waypoint_to_uav: Dict[int, int] = {}

    for uav_id in range(num_uavs):
        sequence_column = f"UAV{uav_id}"

        if sequence_column not in tour_dataframe.columns:
            continue

        sequence_value = row[sequence_column]

        waypoint_ids_in_route = (
            parse_waypoint_ids_from_sequence(sequence_value)
        )

        for wp_id in waypoint_ids_in_route:
            if wp_id in waypoint_to_uav:
                # If a waypoint appears in multiple UAVs,
                # keep the first assignment and warn.
                print(
                    f"[WARNING] {sheet_name}: "
                    f"Waypoint {wp_id} appears in more than one UAV. "
                    f"Keeping first assignment (UAV{waypoint_to_uav[wp_id]})."
                )
                continue

            waypoint_to_uav[wp_id] = uav_id

    # Build rows: one per waypoint in the waypoint file
    rows: List[Dict[str, object]] = []

    for wp_id in waypoint_ids:
        uav = waypoint_to_uav.get(wp_id)

        rows.append(
            {
                "Waypoint id": wp_id,
                "UAV/Cluster/K": uav if uav is not None else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def process_tour_file(
    tour_file: Path,
    waypoint_file: Path,
    output_dir: Path,
) -> Path:
    """
    For one tour workbook, create one output workbook with
    waypoint → UAV assignments for every SimRun sheet.
    """
    num_uavs = extract_num_uavs(tour_file)

    print(
        f"\n[ASSIGN] Processing {tour_file.name} "
        f"({num_uavs} UAVs)"
    )

    tour_xls = pd.ExcelFile(tour_file)

    output_sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name in tour_xls.sheet_names:
        tour_dataframe = pd.read_excel(
            tour_file,
            sheet_name=sheet_name,
        )

        tour_dataframe.columns = (
            tour_dataframe.columns.astype(str).str.strip()
        )

        waypoint_ids = load_waypoint_ids(
            waypoint_file=waypoint_file,
            sheet_name=sheet_name,
        )

        assignment_df = (
            build_waypoint_uav_assignment_for_sheet(
                tour_dataframe=tour_dataframe,
                waypoint_ids=waypoint_ids,
                num_uavs=num_uavs,
                sheet_name=sheet_name,
            )
        )

        output_sheets[sheet_name] = assignment_df

        print(
            f"[ASSIGN] {sheet_name}: "
            f"{len(assignment_df)} waypoint assignments"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Input: UAVs3_GRID13_1920_16_cluster_ga_sequences.xlsx
    # Output: UAVs3_GRID13_1920_16_cluster_assignment.xlsx
    base_name = tour_file.stem.replace(
        "_sequences",
        "",
    )

    output_file = output_dir / (
        f"{base_name}_cluster_assignment.xlsx"
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:
        for sheet_name, dataframe in output_sheets.items():
            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    return output_file


def main() -> None:
    for num_uavs in range(MIN_UAVS, MAX_UAVS + 1):
        scenario_name = f"UAVs{num_uavs}_GRID{GRID_SIZE}"

        tour_dir = RESULTS_DIR / scenario_name / "tour"
        waypoint_file = (
            WAYPOINTS_DIR / f"{scenario_name}_waypoints.xlsx"
        )
        output_dir = (
            RESULTS_DIR / scenario_name / "cluster_assignment"
        )

        if not tour_dir.exists():
            print(
                f"\n[ASSIGN] Skipping {scenario_name}: "
                f"tour directory not found:\n"
                f"  {tour_dir.resolve()}"
            )
            continue

        if not waypoint_file.exists():
            print(
                f"\n[ASSIGN] Skipping {scenario_name}: "
                f"waypoint file not found:\n"
                f"  {waypoint_file.resolve()}"
            )
            continue

        tour_files = sorted(tour_dir.glob("*.xlsx"))

        if not tour_files:
            print(
                f"\n[ASSIGN] Skipping {scenario_name}: "
                "no Excel tour files found."
            )
            continue

        print(
            f"\n{'=' * 60}\n"
            f"[ASSIGN] Scenario: {scenario_name}\n"
            f"[ASSIGN] Tour files: {len(tour_files)}\n"
            f"{'=' * 60}"
        )

        for tour_file in tour_files:
            try:
                output_file = process_tour_file(
                    tour_file=tour_file,
                    waypoint_file=waypoint_file,
                    output_dir=output_dir,
                )

                print(
                    f"[ASSIGN] Saved: "
                    f"{output_file.resolve()}"
                )

            except Exception as exc:
                print(
                    f"[ASSIGN] Failed: {tour_file.name}\n"
                    f"Reason: {exc}"
                )


if __name__ == "__main__":
    main()