import random

from typing import List, Optional, Tuple, Union

from src.environment import GridEnvironment
from src.models import Depot, UAV, Waypoint
from src.utils import euclidean_distance, travel_time


class GreedyAllocator:
    def __init__(
        self,
        environment: GridEnvironment,
        num_uavs: int,
        uav_speed: float,
        max_flight_time: float,
    ) -> None:
        self.environment = environment
        self.num_uavs = num_uavs
        self.uav_speed = uav_speed
        self.max_flight_time = max_flight_time

        self.uavs: List[UAV] = [UAV(uid=i) for i in range(self.num_uavs)]


    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        """
        Run the greedy assignment algorithm and return the final UAV assignments, unassigned targets, total revenue, and total revenue rate.
        """
        self.reset()
        uavs, unassigned_targets = self._assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate


    def reset(self) -> None:
        """
        Reset all UAVs to their initial state (empty sequence and m_j = 0) for a new simulation run.
        """
        for uav in self.uavs:
            uav.reset()


    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        """
        Build the full tour for a UAV based on its assigned sequence and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + (uav.sequence * uav.m_j) + [self.environment.depot]


    # TODO: I need to fix to cumulative flight time
    def current_tour_time(self, uav: UAV) -> float:
        """
        Compute the current tour flight time for a UAV based on its assigned sequence and number of repetitions.
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self._compute_tour_flight_time(uav.sequence, uav.m_j)


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
        return (uav.m_j * self.uav_speed) / t_j


    def compute_revenue_rate(self, uav: UAV) -> float:
        """
        Compute the revenue rate for a UAV as the product of its monitoring frequency and total revenue.
        """
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)


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


    def _compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        """
        Compute the total flight time of a tour given a sequence of waypoints and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        total = travel_time(depot, first_wp, self.uav_speed)

        for rep in range(m_j):
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
                total += travel_time(wp_a, wp_b, self.uav_speed)

            if rep < m_j - 1:
                total += travel_time(last_wp, first_wp, self.uav_speed)

        total += travel_time(last_wp, depot, self.uav_speed)
        return total


    def _compute_m_j(self, sequence: List[Waypoint]) -> int:
        """
        Compute the maximum number of repetitions (m_j) for a given sequence of waypoints such that the total tour time does not exceed MAX_FLIGHT_TIME.
        """
        if not sequence:
            return 0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        depot_legs_time = (
            travel_time(depot, first_wp, self.uav_speed)
            + travel_time(last_wp, depot, self.uav_speed)
        )

        if depot_legs_time > self.max_flight_time:
            return 0

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:])
        )

        if len(sequence) == 1:
            return 1

        cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)
        repeated_cycle_time = internal_sequence_time + cycle_closure_time

        if repeated_cycle_time <= 0:
            return 1

        remaining_time = self.max_flight_time - depot_legs_time + cycle_closure_time
        max_repetitions = int(remaining_time // repeated_cycle_time)
        return max(max_repetitions, 1)


    def _can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        """
        Check if assigning the target waypoint to the UAV's sequence would still allow for a feasible tour (m_j >= 1).
        """
        trial_sequence = uav.sequence + [target_wp]
        trial_m_j = self._compute_m_j(trial_sequence)
        return trial_m_j >= 1


    def _find_nearest_feasible_target(
        self,
        uav: UAV,
        unassigned_targets: List[Waypoint],
    ) -> Optional[Waypoint]:
        """
        Find the nearest unassigned target waypoint that can be feasibly assigned to the UAV's sequence without exceeding MAX_FLIGHT_TIME.
        """
        if not unassigned_targets:
            return None

        current_location = uav.sequence[-1] if uav.sequence else self.environment.depot
        feasible_targets = [wp for wp in unassigned_targets if self._can_assign_target(uav, wp)]

        if not feasible_targets:
            return None

        min_distance = min(euclidean_distance(current_location, wp) for wp in feasible_targets)
        candidates = [
            wp for wp in feasible_targets
            if euclidean_distance(current_location, wp) == min_distance
        ]
        return random.choice(candidates)


    def _select_uav_with_min_tour_time(self) -> UAV:
        """
        Select the UAV with the minimum current tour time. If multiple UAVs have the same minimum tour time, select one randomly among them.
        """
        min_time = min(self.current_tour_time(uav) for uav in self.uavs)
        candidates = [uav for uav in self.uavs if self.current_tour_time(uav) == min_time]
        return random.choice(candidates)


    def _assign_targets_greedily(self) -> Tuple[List[UAV], List[Waypoint]]:
        """
        Greedily assign target waypoints to UAVs by repeatedly selecting the UAV with the minimum current tour time and assigning it the nearest feasible target waypoint until no more assignments are possible.
        """
        unassigned_targets = self.environment.target_waypoints.copy()

        while unassigned_targets:
            selected_uav = self._select_uav_with_min_tour_time()
            nearest_target = self._find_nearest_feasible_target(selected_uav, unassigned_targets)

            if nearest_target is None:
                break

            selected_uav.sequence.append(nearest_target)
            selected_uav.m_j = self._compute_m_j(selected_uav.sequence)
            unassigned_targets.remove(nearest_target)

        return self.uavs, unassigned_targets
