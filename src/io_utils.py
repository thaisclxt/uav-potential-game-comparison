from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .models import Waypoint


def prepare_scenario_outputs_dirs(
    outputs_dir: Path,
    m: int,
    grid_size: int,
) -> tuple[Path, Path, Path]:
    """
    Create output directories for one algorithm and one scenario.

    Example:
        outputs/cluster_ga/UAVs4_GRID20/
            revenue/
            tour/
    """
    scenario_dir = outputs_dir / f"UAVs{m}_GRID{grid_size}"

    revenue_dir = scenario_dir / "revenue"
    tour_dir = scenario_dir / "tour"

    revenue_dir.mkdir(parents=True, exist_ok=True)
    tour_dir.mkdir(parents=True, exist_ok=True)

    return scenario_dir, revenue_dir, tour_dir


def export_runs_to_excel(
    algorithm_name: str,
    m: int,
    uav_speed: float,
    max_flight_time: float,
    grid_size: int,
    rev_sheets: List[pd.DataFrame],
    seq_sheets: List[pd.DataFrame],
    revenue_dir: Path,
    tour_dir: Path,
) -> Tuple[Path, Path]:
    """Export results for one selected algorithm."""
    rev_path = revenue_dir / (
        f"UAVs{m}_GRID{grid_size}_{algorithm_name}.xlsx"
    )

    seq_path = tour_dir / (
        f"UAVs{m}_GRID{grid_size}_"
        f"{max_flight_time}_{uav_speed}_"
        f"{algorithm_name}_sequences.xlsx"
    )

    with pd.ExcelWriter(rev_path) as writer:
        for index, dataframe in enumerate(rev_sheets, start=1):
            dataframe.to_excel(
                writer,
                sheet_name=f"SimRun{index}",
                index=False,
            )

    with pd.ExcelWriter(seq_path) as writer:
        for index, dataframe in enumerate(seq_sheets, start=1):
            dataframe.to_excel(
                writer,
                sheet_name=f"SimRun{index}",
                index=False,
            )

    return rev_path, seq_path


def load_waypoints_sheet(
    path: str | Path,
    sheet_name: int | str,
) -> List[Waypoint]:
    """Load one worksheet and retain positive-revenue targets only."""
    dataframe = pd.read_excel(
        path,
        sheet_name=sheet_name,
    )

    dataframe = dataframe[dataframe["Revenue"] > 0]

    return [
        Waypoint(
            x=float(row["X"]),
            y=float(row["Y"]),
            wid=int(row["Waypoint"]),
            revenue=float(row["Revenue"]),
        )
        for _, row in dataframe.iterrows()
    ]
