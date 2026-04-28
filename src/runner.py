from pathlib import Path
from typing import List

import pandas as pd

from .config import (
    ProjectConfig,
    SimulationConfig,
    GridConfig,
    UAVConfig,
    WaypointConfig,
)
from .environment import GridEnvironment
from algorithms.greedy import GreedyAllocator
from .io_utils import (
    export_runs_to_excel,
    load_waypoints_sheet,
    prepare_scenario_outputs_dirs,
)
from .models import UAV
from .utils import extract_grid_size, extract_num_uavs


def _build_run_dataframes(
    uavs: List[UAV],
    allocator: GreedyAllocator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Greedy has no negotiation rounds, so store a single row with round 0
    rev_row = {"negotiation_round": 0}
    seq_row = {"negotiation_round": 0}

    for uav in uavs:
        z_j = allocator.compute_revenue_rate(uav)
        seq_ids = [wp.wid for wp in uav.sequence] if uav.sequence else []
        seq_str = "-".join(map(str, seq_ids))

        rev_row[f"UAV{uav.uid}"] = z_j
        seq_row[f"UAV{uav.uid}"] = seq_str
        seq_row[f"m_{uav.uid}"] = uav.m_j

    revenue_df = pd.DataFrame([rev_row])
    sequence_df = pd.DataFrame([seq_row])
    return revenue_df, sequence_df


def run_simulation(
    project_cfg: ProjectConfig,
    sim_cfg: SimulationConfig,
    grid_cfg: GridConfig,
    uav_cfg: UAVConfig,
    wp_cfg: WaypointConfig,
    waypoint_files: List[Path],
) -> None:
    outputs_dir = Path(project_cfg.outputs_dir)

    for wp_file in waypoint_files:
        try:
            n_uavs = extract_num_uavs(str(wp_file))
            grid_size = extract_grid_size(str(wp_file))
        except ValueError as exc:
            print(f"[RUNNER] Skipping {wp_file.name}: {exc}")
            continue

        _, revenue_dir, sequences_dir, _ = prepare_scenario_outputs_dirs(
            outputs_dir=Path(project_cfg.outputs_dir),
            m=n_uavs,
            grid_size=grid_size,
        )

        rev_sheets: List[pd.DataFrame] = []
        seq_sheets: List[pd.DataFrame] = []

        print(f"\n=== Processing waypoint file: {wp_file} (m = {n_uavs}), grid_size = {grid_size} ===")

        xls = pd.ExcelFile(wp_file)
        for sheet_name in xls.sheet_names:
            target_waypoints = load_waypoints_sheet(str(wp_file), sheet_name)

            environment = GridEnvironment(
                project_configuration=project_cfg,
                simulation=sim_cfg,
                target_waypoints=target_waypoints,
                width=grid_size,
                height=grid_size,
                spacing=grid_cfg.spacing,
                depot_location=grid_cfg.depot_location,
                wp_base_revenue=wp_cfg.base_revenue,
                wp_min_revenue=wp_cfg.min_revenue,
                wp_max_revenue=wp_cfg.max_revenue,
                fixed_target_indices=wp_cfg.fixed_target_indices,
            )

            allocator = GreedyAllocator(
                environment=environment,
                num_uavs=n_uavs,
                uav_speed=uav_cfg.speed,
                max_flight_time=uav_cfg.max_flight_time,
            )

            uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

            revenue_df, sequence_df = _build_run_dataframes(
                uavs=uavs,
                allocator=allocator,
            )

            rev_sheets.append(revenue_df)
            seq_sheets.append(sequence_df)

        export_runs_to_excel(
            m=n_uavs,
            uav_speed=uav_cfg.speed,
            max_flight_time=uav_cfg.max_flight_time,
            grid_size=grid_size,
            rev_sheets=rev_sheets,
            seq_sheets=seq_sheets,
            revenue_dir=revenue_dir,
            sequences_dir=sequences_dir,
        )
