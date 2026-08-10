from pathlib import Path

from src.config import load_configuration
from src.runner import run_simulation


def main() -> None:
    settings_path = Path("settings.yaml")

    project_cfg, sim_cfg, grid_cfg, uav_cfg, wp_cfg = load_configuration(settings_path)

    waypoint_files = []
    if sim_cfg.scenario == "excel":
        waypoints_dir = Path(project_cfg.waypoints_dir)
        waypoint_files = sorted(waypoints_dir.rglob("*_waypoints.xlsx"))

    run_simulation(
        project_cfg=project_cfg,
        sim_cfg=sim_cfg,
        grid_cfg=grid_cfg,
        uav_cfg=uav_cfg,
        wp_cfg=wp_cfg,
        waypoint_files=waypoint_files,
    )


if __name__ == "__main__":
    main()
