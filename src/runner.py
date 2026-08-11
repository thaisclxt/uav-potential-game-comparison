from pathlib import Path
from typing import List, Type
import random

import pandas as pd

from .config import (
    ProjectConfig,
    SimulationConfig,
    GridConfig,
    UAVConfig,
    WaypointConfig,
)
from .environment import GridEnvironment
from .io_utils import (
    export_runs_to_excel,
    load_waypoints_sheet,
    prepare_scenario_outputs_dirs,
)
from .models import UAV
from .utils import extract_grid_size, extract_num_uavs

from algorithms.greedy import GreedyAllocator
from algorithms.cluster_ga import ClusterGAAllocator


ALLOCATORS = {
    "greedy": GreedyAllocator,
    "cluster_ga": ClusterGAAllocator,
}


def _build_run_dataframes(
    uavs: List[UAV],
    allocator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev_row = {"negotiation_round": 0}
    seq_row = {"negotiation_round": 0}

    for uav in uavs:
        z_j = allocator.compute_revenue_rate(uav)

        seq_ids = (
            [wp.wid for wp in uav.sequence]
            if uav.sequence
            else []
        )

        rev_row[f"UAV{uav.uid}"] = z_j
        seq_row[f"UAV{uav.uid}"] = "-".join(
            map(str, seq_ids)
        )
        seq_row[f"m_{uav.uid}"] = uav.m_j

    return (
        pd.DataFrame([rev_row]),
        pd.DataFrame([seq_row]),
    )


def _run_one_environment(
    environment: GridEnvironment,
    source_label: str,
    run_label: str,
    n_uavs: int,
    algorithm_name: str,
    revenue_sheets: List[pd.DataFrame],
    tour_sheets: List[pd.DataFrame],
    uav_cfg: UAVConfig,
    random_state: int,
) -> None:
    allocator_class: Type = ALLOCATORS[algorithm_name]

    allocator_kwargs = {
        "environment": environment,
        "num_uavs": n_uavs,
        "uav_speed": uav_cfg.speed,
        "max_flight_time": uav_cfg.max_flight_time,
    }

    # ClusterGA uses a random generator internally.
    # Supplying a run-specific seed makes it reproducible.
    if algorithm_name == "cluster_ga":
        allocator_kwargs["random_state"] = random_state

    allocator = allocator_class(**allocator_kwargs)

    uavs, unassigned, total_revenue, total_revenue_rate = (
        allocator.solve()
    )

    revenue_df, tour_df = _build_run_dataframes(
        uavs=uavs,
        allocator=allocator,
    )

    revenue_sheets.append(revenue_df)
    tour_sheets.append(tour_df)

    print(
        f"[RUNNER][{algorithm_name}] "
        f"{source_label} | {run_label} | "
        f"total revenue = {total_revenue:.2f}, "
        f"total revenue rate = {total_revenue_rate:.2f}, "
        f"unassigned = {len(unassigned)}"
    )


def run_simulation(
    project_cfg: ProjectConfig,
    sim_cfg: SimulationConfig,
    grid_cfg: GridConfig,
    uav_cfg: UAVConfig,
    wp_cfg: WaypointConfig,
    waypoint_files: List[Path],
    algorithm_name: str,
) -> None:
    if algorithm_name not in ALLOCATORS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available algorithms: {list(ALLOCATORS)}"
        )

    base_outputs_dir = Path(project_cfg.outputs_dir)
    algorithm_outputs_dir = base_outputs_dir / algorithm_name

    base_outputs_dir = Path(project_cfg.outputs_dir)
    greedy_outputs_dir = base_outputs_dir / "greedy"
    cluster_ga_outputs_dir = base_outputs_dir / "cluster_ga"

    if sim_cfg.scenario == "excel":
        for wp_file in waypoint_files:
            try:
                n_uavs = extract_num_uavs(str(wp_file))
                grid_size = extract_grid_size(str(wp_file))
            except ValueError as exc:
                print(f"[RUNNER] Skipping {wp_file.name}: {exc}")
                continue

            _, revenue_dir, tour_dir, _ = prepare_scenario_outputs_dirs(
                outputs_dir=algorithm_outputs_dir,
                m=n_uavs,
                grid_size=grid_size,
            )

            revenue_sheets: List[pd.DataFrame] = []
            tour_sheets: List[pd.DataFrame] = []

            print(
                f"\n=== Processing waypoint file: {wp_file} "
                f"(m = {n_uavs}, grid_size = {grid_size}) ==="
            )

            xls = pd.ExcelFile(wp_file)
            for sheet_idx, sheet_name in enumerate(xls.sheet_names, start=1):
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
                    number_targets=wp_cfg.number_targets,
                    revenue_matrix=wp_cfg.revenue_matrix,
                )

                _run_one_environment(
                    environment=environment,
                    source_label=wp_file.name,
                    run_label=f"SimRun{sheet_idx}",
                    n_uavs=n_uavs,
                    algorithm_name=algorithm_name,
                    revenue_sheets=revenue_sheets,
                    tour_sheets=tour_sheets,
                    uav_cfg=uav_cfg,
                    random_state=sim_cfg.seed + sheet_idx,
                )

            export_runs_to_excel(
                m=n_uavs,
                uav_speed=uav_cfg.speed,
                max_flight_time=uav_cfg.max_flight_time,
                grid_size=grid_size,
                rev_sheets=revenue_sheets,
                seq_sheets=tour_sheets,
                revenue_dir=revenue_dir,
                tour_dir=tour_dir,
            )

    else:
        n_uavs = uav_cfg.num_uavs
        grid_size = len(wp_cfg.revenue_matrix) if sim_cfg.scenario == "fixed" else grid_cfg.width

        _, revenue_dir, tour_dir, _ = prepare_scenario_outputs_dirs(
            outputs_dir=algorithm_outputs_dir,
            m=n_uavs,
            grid_size=grid_size,
        )

        revenue_sheets: List[pd.DataFrame] = []
        tour_sheets: List[pd.DataFrame] = []

        print(
            f"\n=== Running {sim_cfg.scenario} simulation "
            f"(n_uavs={n_uavs}, grid={grid_size}x{grid_size}, runs={sim_cfg.number_runs}) ==="
        )

        summary_env = GridEnvironment(
            project_configuration=project_cfg,
            simulation=sim_cfg,
            target_waypoints=None,
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
        summary_env.print_static_summary()

        for run_idx in range(1, sim_cfg.number_runs + 1):
            random.seed(sim_cfg.seed + run_idx)

            environment = GridEnvironment(
                project_configuration=project_cfg,
                simulation=sim_cfg,
                target_waypoints=None,
                width=grid_cfg.width,
                height=grid_cfg.height,
                spacing=grid_cfg.spacing,
                depot_location=grid_cfg.depot_location,
                wp_base_revenue=wp_cfg.base_revenue,
                wp_min_revenue=wp_cfg.min_revenue,
                wp_max_revenue=wp_cfg.max_revenue,
                revenue_matrix=wp_cfg.revenue_matrix,
                number_targets=wp_cfg.number_targets,
            )

            _run_one_environment(
                environment=environment,
                source_label="config-generated",
                run_label=f"SimRun{run_idx}",
                n_uavs=n_uavs,
                algorithm_name=algorithm_name,
                revenue_sheets=revenue_sheets,
                tour_sheets=tour_sheets,
                uav_cfg=uav_cfg,
                random_state=sim_cfg.seed + run_idx,
            )

        export_runs_to_excel(
            m=n_uavs,
            uav_speed=uav_cfg.speed,
            max_flight_time=uav_cfg.max_flight_time,
            grid_size=grid_size,
            rev_sheets=revenue_sheets,
            seq_sheets=tour_sheets,
            revenue_dir=revenue_dir,
            tour_dir=tour_dir,
        )

        print(
            f"[RUNNER] Saved {algorithm_name} outputs to: "
            f"{algorithm_outputs_dir / f'UAVs{n_uavs}_GRID{grid_size}'}"
        )
