import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================
# Configuration
# ============================================================

GRID_WIDTH = 13
GRID_HEIGHT = 13
GRID_SPACING = 5
DEPOT_INDEX = 0

NUM_TARGETS = 22

UAVS = [3, 4, 5, 6, 7, 8, 9, 10]

UAV_SPEED = 10
MAX_FLIGHT_TIME = 30

BASE_REVENUE = 30
MIN_WP_REVENUE = 60
MAX_WP_REVENUE = 600

NUM_SIMULATION_RUNS = 100
BASE_SEED = 42

# ============================================================
# Data models
# ============================================================

@dataclass
class Waypoint:
    x: float
    y: float
    revenue: float = BASE_REVENUE

@dataclass
class UAV:
    uav_id: int
    sequence: List[int] = field(default_factory=list)  # S_j
    m_j: int = 0                                       # number of repetitions

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
        self.depot: Waypoint = self.waypoints[DEPOT_INDEX]
        self.target_waypoints: List[int] = self._select_random_targets()

    def _build_grid(self) -> List[Waypoint]:
        waypoints = []
        for x in range(0, GRID_WIDTH):
            for y in range(0, GRID_HEIGHT):
                wp = Waypoint(x * GRID_SPACING, y * GRID_SPACING)
                waypoints.append(wp)
                print(f"Waypoint {len(waypoints)-1}: ({wp.x}, {wp.y}) with base revenue {wp.revenue}")
        return waypoints

    def _select_random_targets(self) -> List[int]:
        candidate_indices = [i for i in range(len(self.waypoints)) if i != DEPOT_INDEX]
        # Ensure NUM_TARGETS is feasible
        if NUM_TARGETS > len(candidate_indices):
            raise ValueError(
                f"NUM_TARGETS={NUM_TARGETS} exceeds available non-depot waypoints={len(candidate_indices)}"
            )
        return random.sample(candidate_indices, NUM_TARGETS)

    def assign_random_revenues(self) -> None:
        """
        Assign random revenues to the already-selected targets.
        """
        for idx in self.target_waypoints:
            self.waypoints[idx].revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)

# ============================================================
# Utility functions
# ============================================================

def euclidean_distance(wp_a: Waypoint, wp_b: Waypoint) -> float:
    return math.hypot(wp_b.x - wp_a.x, wp_b.y - wp_a.y)

def travel_time(wp_a: Waypoint, wp_b: Waypoint) -> float:
    distance = euclidean_distance(wp_a, wp_b)
    return distance / UAV_SPEED

# ============================================================
# Greedy allocator
# ============================================================

class GreedyAllocator:
    def __init__(self, environment: GridEnvironment, num_uavs: int):
        self.environment = environment
        self.num_uavs = num_uavs
        self.uavs = [UAV(uav_id=i + 1) for i in range(self.num_uavs)]

    # ---------- Time and tour helpers ----------

    def compute_sequence_flight_time(self, sequence: List[int]) -> float:
        """
        Time for a single tour: depot -> sequence once -> depot.
        """
        if not sequence:
            return 0.0

        depot = self.environment.depot
        waypoints = self.environment.waypoints

        total = travel_time(depot, waypoints[sequence[0]])

        for a, b in zip(sequence[:-1], sequence[1:]):
            total += travel_time(waypoints[a], waypoints[b])

        total += travel_time(waypoints[sequence[-1]], depot)

        return total

    def compute_tour_flight_time(self, sequence: List[int], m_j: int) -> float:
        """
        Time for tour = [depot] + sequence repeated m_j times + [depot].
        """
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        waypoints = self.environment.waypoints

        total = 0.0

        # depot -> first waypoint of first repetition
        total += travel_time(depot, waypoints[sequence[0]])

        # within and between repetitions
        for rep in range(m_j):
            for a, b in zip(sequence[:-1], sequence[1:]):
                total += travel_time(waypoints[a], waypoints[b])

            # between repetitions (last of rep -> first of next rep)
            if rep < m_j - 1:
                total += travel_time(waypoints[sequence[-1]], waypoints[sequence[0]])

        # last waypoint -> depot
        total += travel_time(waypoints[sequence[-1]], depot)

        return total

    def compute_m_j(self, sequence: List[int]) -> int:
        """
        Largest m_j such that tour flight time <= MAX_FLIGHT_TIME.
        Uses C0 + m*C1 <= MAX_FLIGHT_TIME.
        """
        if not sequence:
            return 0

        depot = self.environment.depot
        waypoints = self.environment.waypoints

        first = sequence[0]
        last = sequence[-1]

        # One-time legs depot -> first + last -> depot
        C0 = travel_time(depot, waypoints[first]) + travel_time(waypoints[last], depot)

        # Per-cycle time within the sequence + last -> first
        C1 = 0.0
        for a, b in zip(sequence[:-1], sequence[1:]):
            C1 += travel_time(waypoints[a], waypoints[b])
        C1 += travel_time(waypoints[last], waypoints[first])

        # If just depot->first + last->depot already too big, no repetitions
        if C0 > MAX_FLIGHT_TIME:
            return 0

        # Degenerate case
        if C1 == 0.0:
            return 1

        m_max = int((MAX_FLIGHT_TIME - C0) // C1)
        return max(m_max, 1)

    def build_tour(self, uav: UAV) -> List[int]:
        """
        Return index sequence for T_j = [depot] + S_j^m_j + [depot].
        """
        if not uav.sequence or uav.m_j <= 0:
            return [DEPOT_INDEX, DEPOT_INDEX]
        return [DEPOT_INDEX] + (uav.sequence * uav.m_j) + [DEPOT_INDEX]

    def current_tour_time(self, uav: UAV) -> float:
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self.compute_tour_flight_time(uav.sequence, uav.m_j)

    # ---------- Revenue and revenue-rate ----------

    def compute_sequence_revenue(self, sequence: List[int]) -> float:
        return sum(self.environment.waypoints[idx].revenue for idx in sequence)

    def compute_total_revenue(self, uav: UAV) -> float:
        """
        r_j(T) = m_j * sum_{w in S_j} revenue_w
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return uav.m_j * self.compute_sequence_revenue(uav.sequence)

    def compute_monitoring_frequency(self, uav: UAV) -> float:
        """
        f(T_j) ~ m_j * v / T_j
        """
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return (uav.m_j * UAV_SPEED) / t_j

    def compute_revenue_rate(self, uav: UAV) -> float:
        """
        z_j(T) = f(T_j) * r_j(T)
        """
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)

    # ---------- Greedy assignment logic ----------

    def reset(self) -> None:
        for uav in self.uavs:
            uav.reset()

    def can_assign_target(self, uav: UAV, target_idx: int) -> bool:
        """
        Feasible if adding target to sequence yields m_j >= 1.
        """
        trial_sequence = uav.sequence + [target_idx]
        trial_m_j = self.compute_m_j(trial_sequence)
        return trial_m_j >= 1

    def find_nearest_feasible_target(self, uav: UAV, unassigned_targets: List[int]) -> Optional[int]:
        if not unassigned_targets:
            return None

        waypoints = self.environment.waypoints

        current_index = uav.sequence[-1] if uav.sequence else DEPOT_INDEX
        current_wp = waypoints[current_index]

        feasible_targets = [
            t_idx
            for t_idx in unassigned_targets
            if self.can_assign_target(uav, t_idx)
        ]

        if not feasible_targets:
            return None

        return min(
            feasible_targets,
            key=lambda t_idx: euclidean_distance(current_wp, waypoints[t_idx]),
        )

    def select_uav_with_min_tour_time(self) -> UAV:
        """
        Greedy rule: pick UAV with smallest current tour time.
        """
        min_time = min(self.current_tour_time(uav) for uav in self.uavs)

        # If multiple UAVs tie for min time, pick one at random among the ties
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
    unassigned_targets: List[int],
    total_revenue: float,
    total_revenue_rate: float,
    header: str = "Greedy Result",
) -> None:
    print(f"\n=== {header} ===")
    print(f"Number of UAVs (m): {allocator.num_uavs}")
    print(f"Selected targets: {sorted(environment.target_waypoints)}")
    print(f"Total revenue over tours: {total_revenue:.2f}")
    print(f"Total revenue rate (sum_j z_j): {total_revenue_rate:.4f}")

    for uav in uavs:
        seq = uav.sequence
        tour = allocator.build_tour(uav)
        seq_revenue = allocator.compute_sequence_revenue(seq)
        tour_time = allocator.current_tour_time(uav)
        remaining_time = MAX_FLIGHT_TIME - tour_time
        r_j = allocator.compute_total_revenue(uav)
        f_j = allocator.compute_monitoring_frequency(uav)
        z_j = allocator.compute_revenue_rate(uav)

        print(f"\nUAV {uav.uav_id}")
        print(f"  Sequence S_j: {seq if seq else 'No waypoints assigned'}")
        print(f"  m_j (repetitions): {uav.m_j}")
        print(f"  Tour T_j (indices): {tour}")
        print(f"  Sequence revenue (one pass): {seq_revenue}")
        print(f"  Total revenue r_j(T): {r_j:.2f}")
        print(f"  Tour flight time T_j (s): {tour_time:.2f}")
        print(f"  Remaining time (s): {remaining_time:.2f}")
        print(f"  Monitoring frequency f(T_j): {f_j:.6f}")
        print(f"  Revenue rate z_j(T): {z_j:.6f}")

    if unassigned_targets:
        print(f"\nUnassigned targets: {sorted(unassigned_targets)}")
    else:
        print("\nAll targets were assigned to at least one UAV (m_j >= 1).")
        

# ============================================================
# Main: run many simulations and maximize revenue rate
# ============================================================

def run_simulation() -> None:
    # Create a single fixed environment for all runs (same targets and revenues)
    environment = GridEnvironment(seed=BASE_SEED)

    # Ask the professor: how does the 20% high risk revenue woork?
    environment.assign_random_revenues()

    print("\n=== Fixed Environment Summary ===")
    print(f"Total waypoints: {len(environment.waypoints)}")
    print(f"Number of targets: {len(environment.target_waypoints)}\n")
    
    for i in sorted(environment.target_waypoints):
        wp = environment.waypoints[i]
        print(f"Target {i}: ({wp.x}, {wp.y}) with revenue {wp.revenue:.2f}")

    overall_best_revenue_rate = -float("inf")
    overall_best_result = None

    # To store best revenue rate per m
    per_m_best = {}

    for m in UAVS:
        best_total_revenue_rate_for_m = -float("inf")
        best_result_for_m = None

        allocator = GreedyAllocator(environment=environment, num_uavs=m)
        uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

        # Track best revenue rate for this m (maximize)
        if total_revenue_rate > best_total_revenue_rate_for_m:
            best_total_revenue_rate_for_m = total_revenue_rate
            best_result_for_m = (
                allocator,
                uavs,
                unassigned_targets,
                total_revenue,
                total_revenue_rate,
            )

        # Store best revenue rate for this m
        per_m_best[m] = best_total_revenue_rate_for_m

        print(f"\n=== Best result for m = {m} over {NUM_SIMULATION_RUNS} runs ===")
        if best_result_for_m is None:
            print("  No feasible solution found.")
        else:
            allocator, uavs, unassigned_targets, total_revenue, total_revenue_rate = best_result_for_m
            print(f"  Best total revenue rate: {best_total_revenue_rate_for_m:.4f}")
            # print_solution(environment, allocator, uavs, unassigned_targets, total_revenue, total_revenue_rate,
            #               header=f"Best Greedy Result for m = {m}")

            # Track overall best across all m (maximize)
            if best_total_revenue_rate_for_m > overall_best_revenue_rate:
                overall_best_revenue_rate = best_total_revenue_rate_for_m
                overall_best_result = (m, environment, allocator, uavs, unassigned_targets,
                                       total_revenue, total_revenue_rate)

    print("\n=== Overall best m (max revenue rate) ===")
    if overall_best_result is None:
        print("No feasible solution found for any m.")
    else:
        best_m, best_env, best_alloc, best_uavs, best_unassigned, best_total_revenue, best_total_revenue_rate = overall_best_result
        print(f"Best m (number of UAVs): {best_m}")
        print(f"Best total revenue rate: {best_total_revenue_rate:.4f}")
        print_solution(
            best_env,
            best_alloc,
            best_uavs,
            best_unassigned,
            best_total_revenue,
            best_total_revenue_rate,
            header="Overall Best Greedy Result (Max Revenue Rate)",
        )

    print("\n=== Summary: best revenue rate per m ===")
    for m in sorted(per_m_best.keys()):
        print(f"m = {m}: best revenue rate = {per_m_best[m]:.4f}")

if __name__ == "__main__":
    run_simulation()