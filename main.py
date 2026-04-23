import math
import random
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import pandas as pd  # NEW: for Excel writing with sheets


# ============================================================
# Configuration
# ============================================================

GRID_WIDTH = 3
GRID_HEIGHT = 3
GRID_SPACING = 1
DEPOT_LOCATION = (0, 0)

NUM_TARGETS = 5

USE_FIXED_TARGETS = True
FIXED_TARGET_INDICES = [2, 3, 5, 6, 7]

UAVS = [1, 2]

UAV_SPEED = 10
MAX_FLIGHT_TIME = 2

BASE_REVENUE = 30
MIN_WP_REVENUE = 60
MAX_WP_REVENUE = 600

NUM_SIMULATION_RUNS = 100
BASE_SEED = 32

OUTPUT_BASE_DIR = "Results/Greedy"


# ============================================================
# Data models
# ============================================================

@dataclass
class Waypoint:
    x: float
    y: float
    wid: int
    revenue: float = BASE_REVENUE


@dataclass
class Depot:
    x: float = DEPOT_LOCATION[0]
    y: float = DEPOT_LOCATION[1]


@dataclass
class UAV:
    uid: int
    sequence: List[Waypoint] = field(default_factory=list)  # S_j
    m_j: int = 0                                            # number of repetitions

    def reset(self) -> None:
        self.sequence.clear()
        self.m_j = 0


# ============================================================
# Environment
# ============================================================

class GridEnvironment:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.waypoints: List[Waypoint] = self._build_grid()
        self.depot: Depot = Depot()
        self.target_waypoints: List[Waypoint] = self._select_random_targets()

    def print_summary(self) -> None:
        print(f"Depot at: ({self.depot.x}, {self.depot.y})\n")

        for wp in sorted(self.waypoints, key=lambda w: w.wid):
            print(f"Waypoint {wp.wid}: ({wp.x}, {wp.y})")

        print(f"\nTotal waypoints: {len(self.waypoints)}")
        print(f"Number of targets: {len(self.target_waypoints)}\n")

        for wp in sorted(self.target_waypoints, key=lambda w: w.wid):
            print(f"Waypoint target {wp.wid}: ({wp.x}, {wp.y}) with revenue {wp.revenue:.2f}")

    def _build_grid(self) -> List[Waypoint]:
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

    def _select_random_targets(self) -> List[Waypoint]:
        if USE_FIXED_TARGETS:
            target_indices = FIXED_TARGET_INDICES
            return [self.waypoints[i] for i in target_indices]

        if NUM_TARGETS > len(self.waypoints):
            raise ValueError(f"NUM_TARGETS ({NUM_TARGETS}) exceeds available waypoints ({len(self.waypoints)}).")
        return random.sample(self.waypoints, NUM_TARGETS)

    def assign_random_revenues(self) -> None:
        for wp in self.target_waypoints:
            wp.revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)


# ============================================================
# Utility functions
# ============================================================

def euclidean_distance(a: Waypoint | Depot, b: Waypoint | Depot) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def travel_time(a: Waypoint | Depot, b: Waypoint | Depot) -> float:
    distance = euclidean_distance(a, b)
    return distance / UAV_SPEED


# ============================================================
# Greedy allocator
# ============================================================

class GreedyAllocator:
    def __init__(self, environment: GridEnvironment, num_uavs: int):
        self.environment = environment
        self.num_uavs = num_uavs
        self.uavs = [UAV(uid=i + 1) for i in range(self.num_uavs)]

    # ---------- Time and tour helpers ----------

    def compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
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
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + (uav.sequence * uav.m_j) + [self.environment.depot]

    def current_tour_time(self, uav: UAV) -> float:
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self.compute_tour_flight_time(uav.sequence, uav.m_j)

    # ---------- Revenue and revenue-rate ----------

    def compute_sequence_revenue(self, sequence: List[Waypoint]) -> float:
        return sum(wp.revenue for wp in sequence)

    def compute_total_revenue(self, uav: UAV) -> float:
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return uav.m_j * self.compute_sequence_revenue(uav.sequence)

    def compute_monitoring_frequency(self, uav: UAV) -> float:
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return (uav.m_j * UAV_SPEED) / t_j

    def compute_revenue_rate(self, uav: UAV) -> float:
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)

    # ---------- Greedy assignment logic ----------

    def reset(self) -> None:
        for uav in self.uavs:
            uav.reset()

    def can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        trial_sequence = uav.sequence + [target_wp]
        trial_m_j = self.compute_m_j(trial_sequence)
        return trial_m_j >= 1

    def find_nearest_feasible_target(self, uav: UAV, unassigned_targets: List[Waypoint]) -> Optional[Waypoint]:
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
        min_time = min(self.current_tour_time(uav) for uav in self.uavs)
        candidates = [uav for uav in self.uavs if self.current_tour_time(uav) == min_time]
        return random.choice(candidates)

    def assign_targets_greedily(self) -> Tuple[List[UAV], List[int]]:
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
        return sum(self.compute_total_revenue(uav) for uav in self.uavs)

    def compute_total_revenue_rate_all(self) -> float:
        return sum(self.compute_revenue_rate(uav) for uav in self.uavs)

    def solve(self) -> Tuple[List[UAV], List[int], float, float]:
        self.reset()
        uavs, unassigned_targets = self.assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate


# ============================================================
# Reporting
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
    date_str = datetime.now().strftime("%Y-%m-%d")

    revenue_base = os.path.join(base_dir, "revenue", date_str)
    tours_base = os.path.join(base_dir, "sequences", date_str)

    os.makedirs(revenue_base, exist_ok=True)
    os.makedirs(tours_base, exist_ok=True)

    existing = [
        d for d in os.listdir(revenue_base)
        if os.path.isdir(os.path.join(revenue_base, d)) and d.startswith("simulation_")
    ]
    sim_idx = len(existing) + 1

    revenue_dir = os.path.join(revenue_base, f"simulation_{sim_idx}")
    tours_dir = os.path.join(tours_base, f"simulation_{sim_idx}")

    os.makedirs(revenue_dir, exist_ok=True)
    os.makedirs(tours_dir, exist_ok=True)

    return revenue_dir, tours_dir


def export_runs_to_excel_per_m(
    m: int,
    rev_sheets: List[pd.DataFrame],
    seq_sheets: List[pd.DataFrame],
    revenue_dir: str,
    tours_dir: str,
) -> Tuple[str, str]:
    rev_path = os.path.join(
        revenue_dir,
        f"UAVs{m}_GRID{GRID_WIDTH}_Greedy.xlsx"
    )
    seq_path = os.path.join(
        tours_dir,
        f"UAVs{m}_GRID{GRID_WIDTH}_{MAX_FLIGHT_TIME}_{UAV_SPEED}_Greedy_sequences.xlsx"
    )

    with pd.ExcelWriter(rev_path) as writer:
        for idx, df in enumerate(rev_sheets, start=1):
            sheet_name = f"SimRun{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index="Round")

    with pd.ExcelWriter(seq_path) as writer:
        for idx, df in enumerate(seq_sheets, start=1):
            sheet_name = f"SimRun{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index="Round")

    return rev_path, seq_path


# ============================================================
# Main: run many simulations and save per-UAV rows in Excel
# ============================================================

def run_simulation() -> None:
    environment = GridEnvironment(seed=BASE_SEED)
    environment.assign_random_revenues()
    environment.print_summary()

    overall_best_revenue_rate = -float("inf")
    overall_best_result = None
    per_m_best = {}

    for m in UAVS:
        best_total_revenue_rate_for_m = -float("inf")
        best_result_for_m = None

        rev_sheets: List[pd.DataFrame] = []
        seq_sheets: List[pd.DataFrame] = []

        for run_idx in range(1, NUM_SIMULATION_RUNS + 1):
            allocator = GreedyAllocator(environment=environment, num_uavs=m)
            uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

            # build one row per UAV, greedy has no negotiation rounds -> use 0
            rev_row = {"negotiation_round": 0}
            seq_row = {"negotiation_round": 0}

            for uav in uavs:
                z_j = allocator.compute_revenue_rate(uav)
                seq_ids = [wp.wid for wp in uav.sequence] if uav.sequence else []
                seq_str = "-".join(map(str, seq_ids))

                rev_row[f"UAV{uav.uid}"] = z_j

                seq_row[f"UAV{uav.uid}"] = seq_str
                seq_row[f"m_{uav.uid}"] = uav.m_j

            df_rev = pd.DataFrame([rev_row])
            df_seq = pd.DataFrame([seq_row])

            rev_sheets.append(df_rev)
            seq_sheets.append(df_seq)

            if total_revenue_rate > best_total_revenue_rate_for_m:
                best_total_revenue_rate_for_m = total_revenue_rate
                best_result_for_m = (
                    allocator,
                    uavs,
                    unassigned_targets,
                    total_revenue,
                    total_revenue_rate,
                )

        per_m_best[m] = best_total_revenue_rate_for_m

        revenue_dir, tours_dir = prepare_output_dirs(OUTPUT_BASE_DIR)
        revenue_file, tours_file = export_runs_to_excel_per_m(
            m=m,
            rev_sheets=rev_sheets,
            seq_sheets=seq_sheets,
            revenue_dir=revenue_dir,
            tours_dir=tours_dir,
        )

        print(f"\nSaved revenue Excel:   {revenue_file}")
        print(f"Saved sequences Excel: {tours_file}")

        if best_result_for_m is None:
            print("No feasible solution found.")
        else:
            allocator, uavs, unassigned_targets, total_revenue, total_revenue_rate = best_result_for_m
            print_solution(
                environment,
                allocator,
                uavs,
                unassigned_targets,
                total_revenue,
                total_revenue_rate,
                header=f"Best Greedy Result for m = {m}"
            )

            if best_total_revenue_rate_for_m > overall_best_revenue_rate:
                overall_best_revenue_rate = best_total_revenue_rate_for_m
                overall_best_result = (
                    m,
                    environment,
                    allocator,
                    uavs,
                    unassigned_targets,
                    total_revenue,
                    total_revenue_rate,
                )


if __name__ == "__main__":
    run_simulation()