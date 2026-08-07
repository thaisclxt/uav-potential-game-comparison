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
        number_targets: int,
        target_waypoints: Optional[List[Waypoint]] = None,
        revenue_matrix: Optional[list[list[float]]] = None,
    ) -> None:
        self.project_configuration = project_configuration
        self.simulation = simulation

        self.spacing = spacing
        self.depot = Depot(location=depot_location)

        self.wp_base_revenue = wp_base_revenue
        self.wp_min_revenue = wp_min_revenue
        self.wp_max_revenue = wp_max_revenue

        self.number_targets = number_targets

        # Store matrix, but its usage depends on scenario
        self.revenue_matrix = revenue_matrix or []

        # Derive grid size from matrix for fixed scenario
        if simulation.scenario == "fixed" and self.revenue_matrix:
            self.height = len(self.revenue_matrix)
            self.width = len(self.revenue_matrix[0])
        else:
            self.width = width
            self.height = height

        # Build grid once (used for fixed and random scenarios)
        self.waypoints = self.build_grid()

        if target_waypoints is not None:
            # EXCEL: use waypoints from Excel, keep only positive revenue
            self.target_waypoints = [wp for wp in target_waypoints if wp.revenue > 0]
            self.waypoints = []
            # Revenues already set from Excel, do nothing
        elif simulation.scenario == "fixed":
            # FIXED: all non-depot points are targets, use revenue_matrix
            if not self.revenue_matrix:
                raise ValueError("scenario='fixed' requires a non-empty revenue_matrix")
            self.target_waypoints = self.waypoints.copy()
            self.assign_fixed_revenues()
            # Filter out zero-revenue waypoints
            self.target_waypoints = [wp for wp in self.target_waypoints if wp.revenue > 0]
        elif simulation.scenario == "random":
            # RANDOM: pick random targets, assign random revenues
            self.target_waypoints = self.generate_target_waypoints()
            self.assign_random_revenues()
            # Filter out zero-revenue waypoints (in case min_revenue is 0)
            self.target_waypoints = [wp for wp in self.target_waypoints if wp.revenue > 0]
        else:
            raise ValueError(f"Unknown scenario: {simulation.scenario}")

    def build_grid(self) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        wid = 0
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == (self.depot.x, self.depot.y):
                    continue
                x = i * self.spacing
                y = j * self.spacing
                wp = Waypoint(
                    x=x,
                    y=y,
                    wid=wid,
                    revenue=self.wp_base_revenue,
                )
                # Store grid indices for matrix lookup
                wp.grid_x = i
                wp.grid_y = j
                waypoints.append(wp)
                wid += 1
        return waypoints

    def generate_target_waypoints(self) -> List[Waypoint]:
        # Always sample from self.waypoints
        return random.sample(
            self.waypoints,
            self.number_targets,
        )

    def assign_random_revenues(self) -> None:
        for wp in self.target_waypoints:
            wp.revenue = random.uniform(
                self.wp_min_revenue,
                self.wp_max_revenue,
            )

    def assign_fixed_revenues(self) -> None:
        """
        Matrix indexing:
          row = grid_x, col = grid_y
          revenue_matrix[row][col]

        Depot at (0,0) is excluded from waypoints, so we never access (0,0).
        """
        for wp in self.waypoints:
            row = wp.grid_x
            col = wp.grid_y
            wp.revenue = self.revenue_matrix[row][col]

    def print_static_summary(self) -> None:
        print(f"Depot at: ({self.depot.x}, {self.depot.y})\n")

        for wp in sorted(self.waypoints, key=lambda w: w.wid):
            print(
                f"Waypoint {wp.wid}: "
                f"({wp.x}, {wp.y}) "
                f"Revenue={wp.revenue}"
            )

        print(f"\nTotal waypoints: {len(self.waypoints)}")
        print(f"Number of targets: {len(self.target_waypoints)}\n")