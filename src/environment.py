import random

from typing import List, Optional

from .config import ProjectConfig, SimulationConfig
from .models import Depot, Waypoint


class GridEnvironment:
    def __init__(
        self,
        project_configuration: ProjectConfig,
        simulation: SimulationConfig,
        width: int,
        height: int,
        spacing: float,
        depot_location: tuple[float, float],
        wp_base_revenue: float,
        wp_min_revenue: float,
        wp_max_revenue: float,
        target_waypoints: Optional[List[Waypoint]] = None,
        fixed_target_indices: Optional[list[int]] = None,
    ) -> None:
        self.project_configuration = project_configuration
        self.simulation = simulation
        self.width = width
        self.height = height
        self.spacing = spacing
        self.depot = Depot(location=depot_location)

        self.wp_base_revenue = wp_base_revenue
        self.wp_min_revenue = wp_min_revenue
        self.wp_max_revenue = wp_max_revenue
        self.fixed_target_indices = fixed_target_indices or []

        if target_waypoints is not None:
            self.target_waypoints = target_waypoints
            self.waypoints: List[Waypoint] = []
        else:
            self.waypoints = self.build_grid()
            self.target_waypoints = self.generate_target_waypoints()
            self.assign_random_revenues()


    def build_grid(self) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        wid = 0
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) == (self.depot.x, self.depot.y):
                    continue
                x = j * self.spacing
                y = i * self.spacing
                waypoints.append(Waypoint(x=x, y=y, wid=wid, revenue=self.wp_base_revenue))
                wid += 1
        return waypoints


    def generate_target_waypoints(self) -> List[Waypoint]:
        if self.simulation.generate_random_targets:
            return random.sample(self.waypoints, self.simulation.number_targets)
        return [self.waypoints[i] for i in self.fixed_target_indices]


    def assign_random_revenues(self) -> None:
        for wp in self.target_waypoints:
            wp.revenue = random.uniform(self.wp_min_revenue, self.wp_max_revenue)


    def print_static_summary(self) -> None:
        print(f"Depot at: ({self.depot.x}, {self.depot.y})\n")

        for wp in sorted(self.waypoints, key=lambda w: w.wid):
            print(f"Waypoint {wp.wid}: ({wp.x}, {wp.y})")

        print(f"\nTotal waypoints: {len(self.waypoints)}")
        print(f"Number of targets: {len(self.target_waypoints)}\n")
