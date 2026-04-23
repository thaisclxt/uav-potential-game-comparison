import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

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
        # Only for testing: use fixed target indices instead of random sampling
        if USE_FIXED_TARGETS:
            target_indices = FIXED_TARGET_INDICES
            return [self.waypoints[i] for i in target_indices]

        # Ensure NUM_TARGETS is feasible
        if NUM_TARGETS > len(self.waypoints):
            raise ValueError(f"NUM_TARGETS ({NUM_TARGETS}) exceeds available waypoints ({len(self.waypoints)}).")
        return random.sample(self.waypoints, NUM_TARGETS)
    


    def assign_random_revenues(self) -> None:
        """
        Assign random revenues to the already-selected targets.
        """
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
        """
        Time for tour = [depot] + sequence repeated m_j times + [depot].
        """
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot

        first_wp = sequence[0]
        last_wp = sequence[-1]

        # depot -> first waypoint
        total = travel_time(depot, first_wp)

        for rep in range(m_j):
            # within sequence
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
                total += travel_time(wp_a, wp_b)

            # between repetitions (last of rep -> first of next rep)
            if rep < m_j - 1:
                total += travel_time(last_wp, first_wp)

        # last waypoint -> depot
        total += travel_time(last_wp, depot)

        return total

    def compute_m_j(self, sequence: List[Waypoint]) -> int:
        """
        Compute the maximum m_j such that the tour time does not exceed MAX_FLIGHT_TIME.
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
        
        # Time to close the cycle (last waypoint back to first)
        internal_sequence_time += travel_time(last_wp, first_wp)

        # If just depot->first + last->depot already exceeds max time, no repetitions are possible
        if depot_legs_time > MAX_FLIGHT_TIME:
            return 0
        
        if internal_sequence_time == 0.0:
            return 1

        remaining_time = MAX_FLIGHT_TIME - depot_legs_time

        max_repetitions = int(remaining_time // internal_sequence_time)
        return max(max_repetitions, 1)
        

    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        """
        Return index sequence for T_j = [depot] + S_j^m_j + [depot].
        """
        if not uav.sequence or uav.m_j <= 0:
            # Just the depot if no waypoints assigned
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

    def can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        """
        Feasible if adding target to sequence yields m_j >= 1.
        """
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

        # If multiple targets tie for nearest, pick one at random among the ties
        candidates = [wp for wp in feasible_targets if euclidean_distance(current_location, wp) == min_distance]
        return random.choice(candidates)

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
    unassigned_targets: List[Waypoint],
    total_revenue: float,
    total_revenue_rate: float,
    header: str = "Greedy Result",
) -> None:
    print(f"\n=== {header} ===")
    print(f"Number of UAVs (m): {allocator.num_uavs}")
    print(f"Selected targets: {sorted([wp.wid for wp in environment.target_waypoints])}")
    # print(f"Total revenue over tours: {total_revenue:.2f}")
    # print(f"Total revenue rate (sum_j z_j): {total_revenue_rate:.4f}")

    for uav in uavs:
        seq = uav.sequence
        tour = allocator.build_tour(uav)
        seq_revenue = allocator.compute_sequence_revenue(seq)
        tour_time = allocator.current_tour_time(uav)
        remaining_time = MAX_FLIGHT_TIME - tour_time
        r_j = allocator.compute_total_revenue(uav)
        f_j = allocator.compute_monitoring_frequency(uav)
        z_j = allocator.compute_revenue_rate(uav)

        print(f"\nUAV {uav.uid}")
        print(f"  Sequence S_j: {[wp.wid for wp in seq] if seq else 'No waypoints assigned'}")
        print(f"  m_j (repetitions): {uav.m_j}")
        print(f"  Tour T_j (indices): {[wp.wid if isinstance(wp, Waypoint) else environment.depot for wp in tour]}")
        # print(f"  Sequence revenue (one pass): {seq_revenue}")
        # print(f"  Total revenue r_j(T): {r_j:.2f}")
        print(f"  Tour flight time T_j (s): {tour_time:.2f}")
        print(f"  Remaining time (s): {remaining_time:.2f}")
        # print(f"  Monitoring frequency f(T_j): {f_j:.6f}")
        # print(f"  Revenue rate z_j(T): {z_j:.6f}")

    if unassigned_targets:
        print(f"\nUnassigned targets: {sorted([wp.wid for wp in unassigned_targets])}")
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

    # Print environment summary (targets and their revenues)
    environment.print_summary()

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

        # print(f"\n=== Best result for m = {m} over {NUM_SIMULATION_RUNS} runs ===")
        if best_result_for_m is None:
            print("  No feasible solution found.")
        else:
            allocator, uavs, unassigned_targets, total_revenue, total_revenue_rate = best_result_for_m
            # print(f"  Best total revenue rate: {best_total_revenue_rate_for_m:.4f}")
            print_solution(environment, allocator, uavs, unassigned_targets, total_revenue, total_revenue_rate,
                          header=f"Best Greedy Result for m = {m}")

            # Track overall best across all m (maximize)
            if best_total_revenue_rate_for_m > overall_best_revenue_rate:
                overall_best_revenue_rate = best_total_revenue_rate_for_m
                overall_best_result = (m, environment, allocator, uavs, unassigned_targets,
                                       total_revenue, total_revenue_rate)

    # print("\n=== Overall best m (max revenue rate) ===")
    # if overall_best_result is None:
    #     print("No feasible solution found for any m.")
    # else:
    #     best_m, best_env, best_alloc, best_uavs, best_unassigned, best_total_revenue, best_total_revenue_rate = overall_best_result
    #     print(f"Best m (number of UAVs): {best_m}")
    #     print(f"Best total revenue rate: {best_total_revenue_rate:.4f}")
    #     print_solution(
    #         best_env,
    #         best_alloc,
    #         best_uavs,
    #         best_unassigned,
    #         best_total_revenue,
    #         best_total_revenue_rate,
    #         header="Overall Best Greedy Result (Max Revenue Rate)",
    #     )

    # print("\n=== Summary: best revenue rate per m ===")
    # for m in sorted(per_m_best.keys()):
    #     print(f"m = {m}: best revenue rate = {per_m_best[m]:.4f}")

if __name__ == "__main__":
    run_simulation()