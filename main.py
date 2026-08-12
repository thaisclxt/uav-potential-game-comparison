import argparse
import time
from pathlib import Path

from src.config import load_configuration
from src.runner import run_simulation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a UAV waypoint-allocation simulation."
    )

    parser.add_argument(
        "--algorithm",
        choices=["greedy", "cluster_ga"],
        required=True,
    )

    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.yaml"),
    )

    return parser.parse_args()


def main() -> None:
    full_program_start = time.perf_counter()

    args = parse_arguments()

    (
        project_cfg,
        sim_cfg,
        grid_cfg,
        uav_cfg,
        wp_cfg,
    ) = load_configuration(args.settings)

    waypoint_files = []

    if sim_cfg.scenario == "excel":
        waypoints_dir = Path(project_cfg.waypoints_dir)

        waypoint_files = sorted(
            waypoints_dir.rglob("*_waypoints.xlsx")
        )

    run_simulation(
        project_cfg=project_cfg,
        sim_cfg=sim_cfg,
        grid_cfg=grid_cfg,
        uav_cfg=uav_cfg,
        wp_cfg=wp_cfg,
        waypoint_files=waypoint_files,
        algorithm_name=args.algorithm,
    )

    full_program_elapsed = time.perf_counter() - full_program_start

    print(
        f"\n[MAIN] Full program runtime for {args.algorithm}: "
        f"{full_program_elapsed:.2f} seconds "
        f"({full_program_elapsed / 60:.2f} minutes)."
    )


if __name__ == "__main__":
    main()