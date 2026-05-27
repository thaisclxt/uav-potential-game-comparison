from pathlib import Path

from src.config import load_configuration
from src.runner import run_simulation
from src.visualization import generate_uav_contribution_boxplots


def main() -> None:
    settings_path = Path("settings.yaml")

    project_cfg, sim_cfg, grid_cfg, uav_cfg, wp_cfg = load_configuration(settings_path)

    waypoint_files = []
    if sim_cfg.use_external_waypoints:
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

    generate_uav_contribution_boxplots(Path(project_cfg.outputs_dir) / "greedy")
    generate_uav_contribution_boxplots(Path(project_cfg.outputs_dir) / "cluster_ga")


if __name__ == "__main__":
    main()
