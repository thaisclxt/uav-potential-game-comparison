import argparse
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
        help="Waypoint-allocation algorithm to run.",
    )

    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.yaml"),
        help="Path to the simulation settings YAML file.",
    )

    return parser.parse_args()


def main() -> None:
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


if __name__ == "__main__":
    main()