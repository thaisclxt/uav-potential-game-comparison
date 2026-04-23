import math, random, os, re
import pandas as pd

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union


# ============================================================
# Configuration
# 
# Note: USE_EXTERNAL_WAYPOINTS = True means the simulation will load waypoints from generated Excel files in the WAYPOINTS_ROOT directory.
# If you want to run the simulation with a default grid environment set USE_EXTERNAL_WAYPOINTS = False and adjust the grid parameters as needed.
# ============================================================

USE_EXTERNAL_WAYPOINTS = True
MAX_FLIGHT_TIME = 1920 # TODO: confirm that this is the parameter used
UAV_SPEED = 16 # TODO: confirm that this is the parameter used

DEPOT_LOCATION = (0, 0)                     # Depot is fixed at the origin (0, 0)

BASE_REVENUE = 30 # TODO: confirm that this is the parameter used
BASE_SEED = 32

OUTPUT_BASE_DIR = "results"                 # Base directory for saving results (Excel files)
WAYPOINTS_ROOT = Path("waypoints")          # Root folder containing generated non-overlap waypoint files

# Default grid parameters (used when USE_EXTERNAL_WAYPOINTS is False)
if not USE_EXTERNAL_WAYPOINTS:
    GRID_WIDTH = 3
    GRID_HEIGHT = 3
    GRID_SPACING = 1

    UAV_SPEED = 1
    MAX_FLIGHT_TIME = 20

    MIN_WP_REVENUE = 60
    MAX_WP_REVENUE = 600

    USE_FIXED_TARGETS = True
    FIXED_TARGET_INDICES = [2, 3, 5, 6, 7]

    # If not using fixed targets, define parameters for random target selection and revenue assignment
    if not USE_FIXED_TARGETS:
        NUM_TARGETS = 5


# ============================================================
# Data models
# ============================================================

@dataclass
class Waypoint:
    """
    Represents a target waypoint with coordinates, unique ID, and revenue.
    """
    x: float
    y: float
    wid: int
    revenue: float = BASE_REVENUE


@dataclass
class Depot:
    """
    Represents the depot location.
    """
    x: float = DEPOT_LOCATION[0]
    y: float = DEPOT_LOCATION[1]


@dataclass
class UAV:
    """
    Represents a UAV with a unique ID, assigned sequence of waypoints, and number of repetitions.
    """
    uid: int
    sequence: List[Waypoint] = field(default_factory=list)  # S_j
    m_j: int = 0                                            # number of repetitions

    def reset(self) -> None:
        """
        Reset the UAV's assigned sequence and repetitions for a new simulation run.
        """
        self.sequence.clear()
        self.m_j = 0


# ============================================================
# Environment
# ============================================================

class GridEnvironment:
    """
    Represents the grid environment with a depot and a set of target waypoints.
    Can be provided externally from generated Excel files or built as a default grid.
    """
    def __init__(self, external_targets: Optional[List[Waypoint]] = None):
        self.depot: Depot = Depot()

        if external_targets:
            self.waypoints: List[Waypoint] = external_targets.copy()
            self.target_waypoints: List[Waypoint] = external_targets.copy()
        else:
            random.seed(BASE_SEED)
            self.waypoints: List[Waypoint] = self._build_grid()
            self.target_waypoints: List[Waypoint] = self._select_waypoint_targets()

    def _build_grid(self) -> List[Waypoint]:
        """
        Build a grid of waypoints based on the specified GRID_WIDTH, GRID_HEIGHT, and GRID_SPACING.
        """
        waypoints = []
        wid = 0
        for i in range(GRID_WIDTH):
            for j in range(GRID_HEIGHT):
                if (i, j) == DEPOT_LOCATION:
                    continue
                x = j * GRID_SPACING
                y = i * GRID_SPACING
                waypoints.append(Waypoint(x=x, y=y, wid=wid))
                wid += 1
        return waypoints
    
    def _select_waypoint_targets(self) -> List[Waypoint]:
        """
        Select a fixed subset of waypoints to be the target waypoints.
        """
        if USE_FIXED_TARGETS:
            return [self.waypoints[i] for i in FIXED_TARGET_INDICES]
        return random.sample(self.waypoints, NUM_TARGETS)

    def assign_random_revenues(self) -> None:
        for wp in self.target_waypoints:
            wp.revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)

    def print_summary(self) -> None:
        """
        Print a summary of the environment, including depot location, waypoints, and target waypoints.
        """
        print(f"Depot at: ({self.depot.x}, {self.depot.y})\n")

        for wp in sorted(self.waypoints, key=lambda w: w.wid):
            print(f"Waypoint {wp.wid}: ({wp.x}, {wp.y})")

        print(f"\nTotal waypoints: {len(self.waypoints)}")
        print(f"Number of targets: {len(self.target_waypoints)}\n")

        for wp in sorted(self.target_waypoints, key=lambda w: w.wid):
            print(f"Waypoint target {wp.wid}: ({wp.x}, {wp.y}) with revenue {wp.revenue:.2f}")


# ============================================================
# Utility functions
# ============================================================

def euclidean_distance(a: Waypoint | Depot, b: Waypoint | Depot) -> float:
    """
    Calculate the Euclidean distance between two points (waypoints or depot).
    """
    return math.hypot(b.x - a.x, b.y - a.y)


def travel_time(a: Waypoint | Depot, b: Waypoint | Depot) -> float:
    """
    Calculate the travel time between two points (waypoints or depot) based on the UAV speed.
    """
    return euclidean_distance(a, b) / UAV_SPEED


def extract_num_uavs(file_path: str) -> int:
    """
    Extract the number of UAVs (m) from the filename using a regular expression.
    (e.g., from UAVs10_GRID13_waypoints.xlsx it will extract 10)
    """
    match = re.search(r"UAVs(\d+)", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not infer number of UAVs from filename: {file_path}")
    return int(match.group(1))

def extract_grid_size(file_path: str) -> int:
    """
    Extract the grid size from the filename using a regular expression.
    (e.g., from UAVs10_GRID13_waypoints.xlsx it will extract 13)
    """
    match = re.search(r"GRID(\d+)", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not infer grid size from filename: {file_path}")
    return int(match.group(1))


def load_waypoints_sheet(waypoints_file: str, sheet_name: str) -> List[Waypoint]:
    """
    Load waypoints from a specific sheet in the given Excel file. The sheet must contain columns:
    - "Waypoint": unique ID of the waypoint
    - "Revenue": revenue value for the waypoint
    - "X": x-coordinate of the waypoint
    - "Y": y-coordinate of the waypoint
    """
    df = pd.read_excel(waypoints_file, sheet_name=sheet_name)

    required_cols = {"Waypoint", "Revenue", "X", "Y"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Sheet {sheet_name} in {waypoints_file} must contain columns {required_cols}, "
            f"but has {list(df.columns)}"
        )

    targets: List[Waypoint] = []
    for _, row in df.iterrows():
        targets.append(
            Waypoint(
                x=float(row["X"]),
                y=float(row["Y"]),
                wid=int(row["Waypoint"]),
                revenue=float(row["Revenue"]),
            )
        )

    return targets


# ============================================================
# Greedy allocator
# ============================================================

class GreedyAllocator:
    """
    Implements a greedy algorithm to assign target waypoints to UAVs while maximizing revenue rate.
    """
    def __init__(self, environment: GridEnvironment, num_uavs: int):
        self.environment = environment
        self.num_uavs = num_uavs
        self.uavs = [UAV(uid=i) for i in range(num_uavs)]

    # ---------- Time and tour helpers ----------

    def compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        """
        Compute the total flight time of a tour given a sequence of waypoints and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        total = travel_time(depot, first_wp)

        for rep in range(m_j):
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
                total += travel_time(wp_a, wp_b)

            if rep < m_j - 1:
                total += travel_time(last_wp, first_wp)

        total += travel_time(last_wp, depot)
        return total

    def compute_m_j(self, sequence: List[Waypoint]) -> int:
        """
        Compute the maximum number of repetitions (m_j) for a given sequence of waypoints such that the total tour time does not exceed MAX_FLIGHT_TIME.
        """
        if not sequence:
            return 0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        depot_legs_time = travel_time(depot, first_wp) + travel_time(last_wp, depot)

        internal_sequence_time = 0.0
        for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
            internal_sequence_time += travel_time(wp_a, wp_b)

        internal_sequence_time += travel_time(last_wp, first_wp)

        if depot_legs_time > MAX_FLIGHT_TIME:
            return 0

        if internal_sequence_time == 0.0:
            return 1

        remaining_time = MAX_FLIGHT_TIME - depot_legs_time
        max_repetitions = int(remaining_time // internal_sequence_time)
        return max(max_repetitions, 1)

    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        """
        Build the full tour for a UAV based on its assigned sequence and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + (uav.sequence * uav.m_j) + [self.environment.depot]

    def current_tour_time(self, uav: UAV) -> float:
        """
        Compute the current tour flight time for a UAV based on its assigned sequence and number of repetitions.
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self.compute_tour_flight_time(uav.sequence, uav.m_j)

    # ---------- Revenue and revenue-rate ----------

    def compute_sequence_revenue(self, sequence: List[Waypoint]) -> float:
        """
        Compute the total revenue of a sequence of waypoints by summing their individual revenues.
        """
        return sum(wp.revenue for wp in sequence)

    def compute_total_revenue(self, uav: UAV) -> float:
        """
        Compute the total revenue for a UAV based on its assigned sequence and number of repetitions.
        The total revenue is the revenue of the sequence multiplied by the number of repetitions (m_j).
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return uav.m_j * self.compute_sequence_revenue(uav.sequence)

    def compute_monitoring_frequency(self, uav: UAV) -> float:
        """
        Compute the monitoring frequency for a UAV based on its assigned sequence and number of repetitions.
        """
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return (uav.m_j * UAV_SPEED) / t_j

    def compute_revenue_rate(self, uav: UAV) -> float:
        """
        Compute the revenue rate for a UAV as the product of its monitoring frequency and total revenue.
        """
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)

    # ---------- Greedy assignment logic ----------

    def reset(self) -> None:
        """
        Reset all UAVs to their initial state (empty sequence and m_j = 0) for a new simulation run.
        """
        for uav in self.uavs:
            uav.reset()

    def can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        """
        Check if assigning the target waypoint to the UAV's sequence would still allow for a feasible tour (m_j >= 1).
        """
        trial_sequence = uav.sequence + [target_wp]
        trial_m_j = self.compute_m_j(trial_sequence)
        return trial_m_j >= 1

    def find_nearest_feasible_target(self, uav: UAV, unassigned_targets: List[Waypoint]) -> Optional[Waypoint]:
        """
        Find the nearest unassigned target waypoint that can be feasibly assigned to the UAV's sequence without exceeding MAX_FLIGHT_TIME.
        """
        if not unassigned_targets:
            return None

        current_location = uav.sequence[-1] if uav.sequence else self.environment.depot
        feasible_targets = [wp for wp in unassigned_targets if self.can_assign_target(uav, wp)]

        if not feasible_targets:
            return None

        min_distance = min(euclidean_distance(current_location, wp) for wp in feasible_targets)
        candidates = [wp for wp in feasible_targets if euclidean_distance(current_location, wp) == min_distance]
        return random.choice(candidates)

    def select_uav_with_min_tour_time(self) -> UAV:
        """
        Select the UAV with the minimum current tour time. If multiple UAVs have the same minimum tour time, select one randomly among them.
        """
        min_time = min(self.current_tour_time(uav) for uav in self.uavs)
        candidates = [uav for uav in self.uavs if self.current_tour_time(uav) == min_time]
        return random.choice(candidates)

    def assign_targets_greedily(self) -> Tuple[List[UAV], List[Waypoint]]:
        """
        Greedily assign target waypoints to UAVs by repeatedly selecting the UAV with the minimum current tour time and assigning it the nearest feasible target waypoint until no more assignments are possible.
        """
        unassigned_targets = self.environment.target_waypoints.copy()

        while unassigned_targets:
            selected_uav = self.select_uav_with_min_tour_time()
            nearest_target = self.find_nearest_feasible_target(selected_uav, unassigned_targets)

            if nearest_target is None:
                break

            selected_uav.sequence.append(nearest_target)
            selected_uav.m_j = self.compute_m_j(selected_uav.sequence)
            unassigned_targets.remove(nearest_target)

        return self.uavs, unassigned_targets

    # ---------- Aggregate metrics ----------

    def compute_total_revenue_all(self) -> float:
        """
        Compute the total revenue across all UAVs by summing their individual total revenues.
        """
        return sum(self.compute_total_revenue(uav) for uav in self.uavs)

    def compute_total_revenue_rate_all(self) -> float:
        """
        Compute the total revenue rate across all UAVs by summing their individual revenue rates.
        """
        return sum(self.compute_revenue_rate(uav) for uav in self.uavs)

    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        """
        Run the greedy assignment algorithm and return the final UAV assignments, unassigned targets, total revenue, and total revenue rate.
        """
        self.reset()
        uavs, unassigned_targets = self.assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate


# ============================================================
# Output helpers
# ============================================================

def print_solution(
    environment: GridEnvironment,
    allocator: GreedyAllocator,
    uavs: List[UAV],
    unassigned_targets: List[Waypoint],
    total_revenue: float,
    total_revenue_rate: float,
    header: str = "Greedy Result",
) -> None:
    print(f"\n=== {header} ===")
    print(f"Number of UAVs (m): {allocator.num_uavs}")
    print(f"Selected targets: {sorted([wp.wid for wp in environment.target_waypoints])}")

    for uav in uavs:
        seq = uav.sequence
        tour = allocator.build_tour(uav)
        tour_time = allocator.current_tour_time(uav)
        remaining_time = MAX_FLIGHT_TIME - tour_time

        print(f"\nUAV {uav.uid}")
        print(f"  Sequence S_j: {[wp.wid for wp in seq] if seq else 'No waypoints assigned'}")
        print(f"  m_j (repetitions): {uav.m_j}")
        print(f"  Tour T_j (indices): {[wp.wid if isinstance(wp, Waypoint) else environment.depot for wp in tour]}")
        print(f"  Tour flight time T_j (s): {tour_time:.2f}")
        print(f"  Remaining time (s): {remaining_time:.2f}")

    if unassigned_targets:
        print(f"\nUnassigned targets: {sorted([wp.wid for wp in unassigned_targets])}")
    else:
        print("\nAll targets were assigned to at least one UAV (m_j >= 1).")


# ============================================================
# Excel export helpers (one row per UAV, one sheet per simulation)
# ============================================================

def prepare_output_dirs(base_dir: str) -> Tuple[str, str]:
    """
    Prepare the output directories for saving revenue and sequence Excel files. The structure will be:
    - base_dir/
        - revenue/
            - YYYY-MM-DD/
                - simulation_1/
                - simulation_2/
                - ...
        - sequences/
            - YYYY-MM-DD/
                - simulation_1/
                - simulation_2/
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    revenue_base = os.path.join(base_dir, "revenue", date_str)
    sequences_base = os.path.join(base_dir, "sequences", date_str)

    os.makedirs(revenue_base, exist_ok=True)
    os.makedirs(sequences_base, exist_ok=True)

    existing = [
        d for d in os.listdir(revenue_base)
        if os.path.isdir(os.path.join(revenue_base, d)) and d.startswith("simulation_")
    ]
    sim_idx = len(existing) + 1

    revenue_dir = os.path.join(revenue_base, f"simulation_{sim_idx}")
    sequences_dir = os.path.join(sequences_base, f"simulation_{sim_idx}")

    os.makedirs(revenue_dir, exist_ok=True)
    os.makedirs(sequences_dir, exist_ok=True)

    return revenue_dir, sequences_dir


def export_runs_to_excel(
    m: int,
    grid_size: int,
    rev_sheets: List[pd.DataFrame],
    seq_sheets: List[pd.DataFrame],
    revenue_dir: str,
    sequences_dir: str,
) -> Tuple[str, str]:
    """
    Export the revenue and sequence data for multiple simulation runs to separate Excel files.
    Each file will contain one sheet per simulation run, with the sheet name indicating the run number (e.g., "SimRun1", "SimRun2", etc.).
    The revenue file will have columns for each UAV's revenue, while the sequence file will have columns for each UAV's assigned sequence and m_j value.
    """
    rev_path = os.path.join(
        revenue_dir,
        f"UAVs{m}_GRID{grid_size}_Greedy.xlsx"
    )
    seq_path = os.path.join(
        sequences_dir,
        f"UAVs{m}_GRID{grid_size}_{MAX_FLIGHT_TIME}_{UAV_SPEED}_Greedy_sequences.xlsx"
    )

    with pd.ExcelWriter(rev_path) as writer:
        for idx, df in enumerate(rev_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    with pd.ExcelWriter(seq_path) as writer:
        for idx, df in enumerate(seq_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    return rev_path, seq_path


def run_simulation(waypoint_files: List[Path]) -> None:
    """
    Run the simulation using external waypoint files. This function is called when USE_EXTERNAL_WAYPOINTS is True.
    It will process each waypoint file, extract the number of UAVs and grid size, load the waypoints from each sheet, run the greedy allocator, and save the results to Excel files.
    """
    for waypoint_file in waypoint_files:
        m = extract_num_uavs(str(waypoint_file))
        grid_size = extract_grid_size(str(waypoint_file))
        workbook = pd.ExcelFile(waypoint_file)

        rev_sheets: List[pd.DataFrame] = []
        seq_sheets: List[pd.DataFrame] = []

        print(f"\n=== Processing waypoint file: {waypoint_file} (m = {m}), grid_size = {grid_size} ===")

        for sheet_name in workbook.sheet_names:
            generated_targets = load_waypoints_sheet(str(waypoint_file), sheet_name)
            environment = GridEnvironment(external_targets=generated_targets)

            allocator = GreedyAllocator(environment=environment, num_uavs=m)
            uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

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

            rev_sheets.append(pd.DataFrame([rev_row]))
            seq_sheets.append(pd.DataFrame([seq_row]))

        revenue_dir, sequences_dir = prepare_output_dirs(OUTPUT_BASE_DIR)
        revenue_file, sequences_file = export_runs_to_excel(
            m=m,
            grid_size=grid_size,
            rev_sheets=rev_sheets,
            seq_sheets=seq_sheets,
            revenue_dir=revenue_dir,
            sequences_dir=sequences_dir,
        )

        print(f"Saved revenue Excel:   {revenue_file}")
        print(f"Saved sequences Excel: {sequences_file}")


def run_example() -> None:
    """
    Run the example simulation using a default grid environment. This function is called when USE_EXTERNAL_WAYPOINTS is False.
    It will build a default grid environment, assign random revenues to the target waypoints, run the greedy allocator, and print the results to the console.
    """
    environment = GridEnvironment()
    environment.assign_random_revenues()

    print("\n=== Default Grid Environment ===")
    environment.print_summary()

    allocator = GreedyAllocator(environment=environment, num_uavs=2)
    uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

    print_solution(
        environment=environment,
        allocator=allocator,
        uavs=uavs,
        unassigned_targets=unassigned_targets,
        total_revenue=total_revenue,
        total_revenue_rate=total_revenue_rate,
        header="Greedy Result with Default Grid",
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    random.seed(BASE_SEED)

    waypoint_files = sorted(WAYPOINTS_ROOT.rglob("*_waypoints.xlsx"))

    if USE_EXTERNAL_WAYPOINTS:
        if not waypoint_files:
            raise FileNotFoundError(f"No waypoint Excel files found under: {WAYPOINTS_ROOT}")
        run_simulation(waypoint_files)
    else:
        run_example()