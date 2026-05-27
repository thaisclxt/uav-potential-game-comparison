from pathlib import Path
from typing import List
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
from algorithms.greedy import GreedyAllocator
from algorithms.cluster_ga import ClusterGAAllocator
from .io_utils import (
    export_runs_to_excel,
    load_waypoints_sheet,
    prepare_scenario_outputs_dirs,
)
from .models import UAV
from .utils import extract_grid_size, extract_num_uavs


def _build_run_dataframes(
    uavs: List[UAV],
    allocator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    tour_df = pd.DataFrame([seq_row])
    return revenue_df, tour_df


def _run_one_environment(
    environment: GridEnvironment,
    source_label: str,
    run_label: str,
    n_uavs: int,
    greedy_rev_sheets: List[pd.DataFrame],
    greedy_seq_sheets: List[pd.DataFrame],
    clusterga_rev_sheets: List[pd.DataFrame],
    clusterga_tour_sheets: List[pd.DataFrame],
    uav_cfg: UAVConfig,
) -> None:
    greedy_allocator = GreedyAllocator(
        environment=environment,
        num_uavs=n_uavs,
        uav_speed=uav_cfg.speed,
        max_flight_time=uav_cfg.max_flight_time,
    )

    cluster_ga_allocator = ClusterGAAllocator(
        environment=environment,
        num_uavs=n_uavs,
        uav_speed=uav_cfg.speed,
        max_flight_time=uav_cfg.max_flight_time,
    )

    greedy_uavs, greedy_unassigned, greedy_total_revenue, greedy_total_revenue_rate = (
        greedy_allocator.solve()
    )

    greedy_revenue_df, greedy_tour_df = _build_run_dataframes(
        uavs=greedy_uavs,
        allocator=greedy_allocator,
    )
    greedy_rev_sheets.append(greedy_revenue_df)
    greedy_seq_sheets.append(greedy_tour_df)

    print(
        f"[RUNNER][Greedy] {source_label} | {run_label} | "
        f"total revenue = {greedy_total_revenue:.2f}, "
        f"total revenue rate = {greedy_total_revenue_rate:.2f}, "
        f"unassigned = {len(greedy_unassigned)}"
    )

    cluster_uavs, cluster_unassigned, cluster_total_revenue, cluster_total_revenue_rate = (
        cluster_ga_allocator.solve()
    )

    cluster_revenue_df, cluster_tour_df = _build_run_dataframes(
        uavs=cluster_uavs,
        allocator=cluster_ga_allocator,
    )
    clusterga_rev_sheets.append(cluster_revenue_df)
    clusterga_tour_sheets.append(cluster_tour_df)

    print(
        f"[RUNNER][ClusterGA] {source_label} | {run_label} | "
        f"total revenue = {cluster_total_revenue:.2f}, "
        f"total revenue rate = {cluster_total_revenue_rate:.2f}, "
        f"unassigned = {len(cluster_unassigned)}"
    )


def run_simulation(
    project_cfg: ProjectConfig,
    sim_cfg: SimulationConfig,
    grid_cfg: GridConfig,
    uav_cfg: UAVConfig,
    wp_cfg: WaypointConfig,
    waypoint_files: List[Path],
) -> None:
    base_outputs_dir = Path(project_cfg.outputs_dir)
    greedy_outputs_dir = base_outputs_dir / "greedy"
    cluster_ga_outputs_dir = base_outputs_dir / "cluster_ga"

    if sim_cfg.use_external_waypoints:
        for wp_file in waypoint_files:
            try:
                n_uavs = extract_num_uavs(str(wp_file))
                grid_size = extract_grid_size(str(wp_file))
            except ValueError as exc:
                print(f"[RUNNER] Skipping {wp_file.name}: {exc}")
                continue

            _, greedy_revenue_dir, greedy_tour_dir, _ = prepare_scenario_outputs_dirs(
                outputs_dir=greedy_outputs_dir,
                m=n_uavs,
                grid_size=grid_size,
            )

            _, cluster_revenue_dir, cluster_tour_dir, _ = prepare_scenario_outputs_dirs(
                outputs_dir=cluster_ga_outputs_dir,
                m=n_uavs,
                grid_size=grid_size,
            )

            greedy_rev_sheets: List[pd.DataFrame] = []
            greedy_seq_sheets: List[pd.DataFrame] = []
            clusterga_rev_sheets: List[pd.DataFrame] = []
            clusterga_tour_sheets: List[pd.DataFrame] = []

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
                    fixed_target_indices=wp_cfg.fixed_target_indices,
                )

                _run_one_environment(
                    environment=environment,
                    source_label=wp_file.name,
                    run_label=f"SimRun{sheet_idx}",
                    n_uavs=n_uavs,
                    greedy_rev_sheets=greedy_rev_sheets,
                    greedy_seq_sheets=greedy_seq_sheets,
                    clusterga_rev_sheets=clusterga_rev_sheets,
                    clusterga_tour_sheets=clusterga_tour_sheets,
                    uav_cfg=uav_cfg,
                )

            export_runs_to_excel(
                m=n_uavs,
                uav_speed=uav_cfg.speed,
                max_flight_time=uav_cfg.max_flight_time,
                grid_size=grid_size,
                rev_sheets=greedy_rev_sheets,
                seq_sheets=greedy_seq_sheets,
                revenue_dir=greedy_revenue_dir,
                tour_dir=greedy_tour_dir,
            )

            export_runs_to_excel(
                m=n_uavs,
                uav_speed=uav_cfg.speed,
                max_flight_time=uav_cfg.max_flight_time,
                grid_size=grid_size,
                rev_sheets=clusterga_rev_sheets,
                seq_sheets=clusterga_tour_sheets,
                revenue_dir=cluster_revenue_dir,
                tour_dir=cluster_tour_dir,
            )

    else:
        n_uavs = uav_cfg.num_uavs
        grid_size = grid_cfg.width

        _, greedy_revenue_dir, greedy_tour_dir, _ = prepare_scenario_outputs_dirs(
            outputs_dir=greedy_outputs_dir,
            m=n_uavs,
            grid_size=grid_size,
        )

        _, cluster_revenue_dir, cluster_tour_dir, _ = prepare_scenario_outputs_dirs(
            outputs_dir=cluster_ga_outputs_dir,
            m=n_uavs,
            grid_size=grid_size,
        )

        greedy_rev_sheets: List[pd.DataFrame] = []
        greedy_seq_sheets: List[pd.DataFrame] = []
        clusterga_rev_sheets: List[pd.DataFrame] = []
        clusterga_tour_sheets: List[pd.DataFrame] = []

        print(
            f"\n=== Running config-based simulation "
            f"(m = {n_uavs}, grid = {grid_cfg.width}x{grid_cfg.height}, runs = {sim_cfg.number_runs}) ==="
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
            fixed_target_indices=wp_cfg.fixed_target_indices,
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
                fixed_target_indices=wp_cfg.fixed_target_indices,
            )

            _run_one_environment(
                environment=environment,
                source_label="config-generated",
                run_label=f"SimRun{run_idx}",
                n_uavs=n_uavs,
                greedy_rev_sheets=greedy_rev_sheets,
                greedy_seq_sheets=greedy_seq_sheets,
                clusterga_rev_sheets=clusterga_rev_sheets,
                clusterga_tour_sheets=clusterga_tour_sheets,
                uav_cfg=uav_cfg,
            )

        export_runs_to_excel(
            m=n_uavs,
            uav_speed=uav_cfg.speed,
            max_flight_time=uav_cfg.max_flight_time,
            grid_size=grid_size,
            rev_sheets=greedy_rev_sheets,
            seq_sheets=greedy_seq_sheets,
            revenue_dir=greedy_revenue_dir,
            tour_dir=greedy_tour_dir,
        )

        export_runs_to_excel(
            m=n_uavs,
            uav_speed=uav_cfg.speed,
            max_flight_time=uav_cfg.max_flight_time,
            grid_size=grid_size,
            rev_sheets=clusterga_rev_sheets,
            seq_sheets=clusterga_tour_sheets,
            revenue_dir=cluster_revenue_dir,
            tour_dir=cluster_tour_dir,
        )

        print(f"[RUNNER] Saved Greedy outputs to:    {greedy_outputs_dir / f'UAVs{n_uavs}_GRID{grid_size}'}")
        print(f"[RUNNER] Saved ClusterGA outputs to: {cluster_ga_outputs_dir / f'UAVs{n_uavs}_GRID{grid_size}'}")
