from typing import List, Optional, Union

from src.environment import GridEnvironment
from src.models import Depot, UAV, Waypoint
from src.utils import travel_time


class BaseAllocator:
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

        self.uavs: List[UAV] = [UAV(uid=i) for i in range(num_uavs)]

    def reset(self) -> None:
        for uav in self.uavs:
            uav.reset()

    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + uav.sequence * uav.m_j + [self.environment.depot]

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
        tour_time = self.current_tour_time(uav)
        if tour_time <= 0.0:
            return 0.0
        return self.compute_total_revenue(uav) / tour_time

    def compute_total_revenue_all(self) -> float:
        return sum(
            self.compute_total_revenue(uav)
            for uav in self.uavs
        )

    def compute_total_revenue_rate_all(self) -> float:
        return sum(
            self.compute_revenue_rate(uav)
            for uav in self.uavs
        )

    def _compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        outbound_time = travel_time(
            depot,
            first_wp,
            self.uav_speed,
        )

        return_time = travel_time(
            last_wp,
            depot,
            self.uav_speed,
        )

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(
                sequence[:-1],
                sequence[1:],
            )
        )

        cycle_closure_time = 0.0

        if len(sequence) > 1:
            cycle_closure_time = travel_time(
                last_wp,
                first_wp,
                self.uav_speed,
            )

        return (
            outbound_time
            + m_j * internal_sequence_time
            + (m_j - 1) * cycle_closure_time
            + return_time
        )

    def _compute_m_j(self, sequence: List[Waypoint]) -> int:
        if not sequence:
            return 0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        outbound_time = travel_time(
            depot,
            first_wp,
            self.uav_speed,
        )

        return_time = travel_time(
            last_wp,
            depot,
            self.uav_speed,
        )

        fixed_time = outbound_time + return_time

        if fixed_time > self.max_flight_time:
            return 0

        if len(sequence) == 1:
            round_trip = 2.0 * outbound_time

            if round_trip <= 0.0:
                return 0

            if round_trip > self.max_flight_time:
                return 0

            return int(
                self.max_flight_time // round_trip
            )

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(
                sequence[:-1],
                sequence[1:],
            )
        )

        cycle_closure_time = travel_time(
            last_wp,
            first_wp,
            self.uav_speed,
        )

        first_repetition_time = (
            fixed_time
            + internal_sequence_time
        )

        if first_repetition_time > self.max_flight_time:
            return 0

        extra_repetition_time = (
            internal_sequence_time
            + cycle_closure_time
        )

        remaining_time = (
            self.max_flight_time - first_repetition_time
        )

        extra_repetitions = int(remaining_time // extra_repetition_time)

        return 1 + max(extra_repetitions, 0)
