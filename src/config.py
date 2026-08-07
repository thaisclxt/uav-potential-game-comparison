import random, yaml

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    outputs_dir: str
    waypoints_dir: str


@dataclass
class SimulationConfig:
    seed: int
    number_runs: int
    scenario: str

    def __post_init__(self) -> None:
        random.seed(self.seed)


@dataclass
class GridConfig:
    width: int
    height: int
    spacing: float
    depot_location: tuple[float, float]


@dataclass
class UAVConfig:
    num_uavs: int
    speed: float
    max_flight_time: float


@dataclass
class WaypointConfig:
    number_targets: int
    base_revenue: float
    min_revenue: float
    max_revenue: float
    revenue_matrix: list[list[float]]

def load_configuration(path: Path):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    project_d = data.get("project")
    sim_d = data.get("simulation")
    grid_d = data.get("grid")
    depot_d = data.get("depot")
    uav_d = data.get("uav")
    wp_d = data.get("waypoints")

    project_cfg = ProjectConfig(
        outputs_dir=project_d.get("outputs_dir"),
        waypoints_dir=project_d.get("waypoints_dir"),
    )

    sim_cfg = SimulationConfig(
        seed=sim_d.get("seed"),
        number_runs=sim_d.get("number_runs"),
        scenario=sim_d.get("scenario"),
    )

    grid_cfg = GridConfig(
        width=grid_d.get("width"),
        height=grid_d.get("height"),
        spacing=grid_d.get("spacing"),
        depot_location=tuple(depot_d.get("location")),
    )

    uav_cfg = UAVConfig(
        num_uavs=uav_d.get("num_uavs"),
        speed=uav_d.get("speed"),
        max_flight_time=uav_d.get("max_flight_time"),
    )

    wp_cfg = WaypointConfig(
        number_targets=wp_d.get("number_targets"),
        base_revenue=wp_d.get("base_revenue"),
        min_revenue=wp_d.get("min_revenue"),
        max_revenue=wp_d.get("max_revenue"),
        revenue_matrix=wp_d.get("revenue_matrix"),
    )

    return project_cfg, sim_cfg, grid_cfg, uav_cfg, wp_cfg
