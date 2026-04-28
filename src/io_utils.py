import os
import pandas as pd

from pathlib import Path
from typing import List, Tuple

from .models import Waypoint


def prepare_scenario_outputs_dirs(
    outputs_dir: Path,
    m: int,
    grid_size: int,
) -> tuple[Path, Path, Path, Path]:
    scenario_dir = outputs_dir / f"UAVs{m}_GRID{grid_size}"

    revenue_dir = scenario_dir / "revenue"
    sequences_dir = scenario_dir / "sequences"
    visualizations_dir = scenario_dir / "visualizations"

    revenue_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    return scenario_dir, revenue_dir, sequences_dir, visualizations_dir


def export_runs_to_excel(
    m: int,
    uav_speed: float,
    max_flight_time: float,
    grid_size: int,
    rev_sheets: List[pd.DataFrame],
    seq_sheets: List[pd.DataFrame],
    revenue_dir: str,
    sequences_dir: str,
) -> Tuple[str, str]:
    rev_path = os.path.join(
        revenue_dir,
        f"UAVs{m}_GRID{grid_size}_Greedy.xlsx"
    )
    seq_path = os.path.join(
        sequences_dir,
        f"UAVs{m}_GRID{grid_size}_{max_flight_time}_{uav_speed}_Greedy_sequences.xlsx"
    )

    with pd.ExcelWriter(rev_path) as writer:
        for idx, df in enumerate(rev_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    with pd.ExcelWriter(seq_path) as writer:
        for idx, df in enumerate(seq_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    return rev_path, seq_path


def load_waypoints_sheet(path: str | Path, sheet_name: int | str) -> List[Waypoint]:
    """
    Load one sheet from a waypoint Excel file and keep only positive-revenue waypoints.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[df["Revenue"] > 0]

    target_waypoints: List[Waypoint] = []
    for _, row in df.iterrows():
        target_waypoints.append(
            Waypoint(
                x=float(row["X"]),
                y=float(row["Y"]),
                wid=int(row["Waypoint"]),
                revenue=float(row["Revenue"]),
            )
        )
    return target_waypoints
