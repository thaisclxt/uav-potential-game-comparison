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
        self.reset()
        uavs, unassigned_targets = self._assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate

    def reset(self) -> None:
        for uav in self.uavs:
            uav.reset()

    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + (uav.sequence * uav.m_j) + [self.environment.depot]

    def current_tour_time(self, uav: UAV) -> float:
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self._compute_tour_flight_time(uav.sequence, uav.m_j)

    def compute_sequence_revenue(self, sequence: List[Waypoint]) -> float:
        return sum(wp.revenue for wp in sequence)

    def compute_total_revenue(self, uav: UAV) -> float:
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return uav.m_j * self.compute_sequence_revenue(uav.sequence)

    def compute_revenue_rate(self, uav: UAV) -> float:
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return self.compute_total_revenue(uav) / t_j

    def compute_total_revenue_all(self) -> float:
        return sum(self.compute_total_revenue(uav) for uav in self.uavs)

    def compute_total_revenue_rate_all(self) -> float:
        return sum(self.compute_revenue_rate(uav) for uav in self.uavs)

    def _compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        outbound_time = travel_time(depot, first_wp, self.uav_speed)
        return_time = travel_time(last_wp, depot, self.uav_speed)

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:])
        )

        cycle_closure_time = 0.0
        if len(sequence) > 1:
            cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)

        total = (
            outbound_time
            + m_j * internal_sequence_time
            + (m_j - 1) * cycle_closure_time
            + return_time
        )

        return total

    def _compute_m_j(self, sequence: List[Waypoint]) -> int:
        if not sequence:
            return 0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        outbound_time = travel_time(depot, first_wp, self.uav_speed)
        return_time = travel_time(last_wp, depot, self.uav_speed)

        fixed_time = outbound_time + return_time
        if fixed_time > self.max_flight_time:
            return 0

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:])
        )

        if len(sequence) == 1:
            round_trip = 2.0 * outbound_time
            if round_trip > self.max_flight_time:
                return 0
            return int(self.max_flight_time // round_trip)

        cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)
        per_extra_repetition_time = internal_sequence_time + cycle_closure_time

        first_repetition_time = fixed_time + internal_sequence_time
        if first_repetition_time > self.max_flight_time:
            return 0

        remaining_time = self.max_flight_time - first_repetition_time
        extra_repetitions = int(remaining_time // per_extra_repetition_time)

        return 1 + max(extra_repetitions, 0)

    def _can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        trial_sequence = uav.sequence + [target_wp]
        trial_m_j = self._compute_m_j(trial_sequence)
        return trial_m_j >= 1

    def _assign_targets_greedily(self) -> Tuple[List[UAV], List[Waypoint]]:
        unassigned_targets = self.environment.target_waypoints.copy()

        while unassigned_targets:
            made_assignment = False

            for uav in self.uavs:
                if not unassigned_targets:
                    break

                best_target: Optional[Waypoint] = None
                best_rate = -1.0

                # All targets are positive-revenue by construction
                feasible = [
                    wp for wp in unassigned_targets
                    if self._can_assign_target(uav, wp)
                ]

                for target in feasible:
                    trial_sequence = uav.sequence + [target]
                    trial_m_j = self._compute_m_j(trial_sequence)

                    old_sequence = uav.sequence.copy()
                    old_m_j = uav.m_j

                    uav.sequence = trial_sequence
                    uav.m_j = trial_m_j

                    new_rate = self.compute_revenue_rate(uav)

                    uav.sequence = old_sequence
                    uav.m_j = old_m_j

                    if new_rate > best_rate:
                        best_rate = new_rate
                        best_target = target

                if best_target is not None:
                    uav.sequence.append(best_target)
                    uav.m_j = self._compute_m_j(uav.sequence)
                    unassigned_targets.remove(best_target)
                    made_assignment = True

            if not made_assignment:
                break

        return self.uavs, unassigned_targets