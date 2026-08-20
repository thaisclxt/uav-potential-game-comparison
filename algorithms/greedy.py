from typing import List, Optional, Tuple

from algorithms.base_allocator import BaseAllocator
from src.models import UAV, Waypoint


class GreedyAllocator(BaseAllocator):
    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        self.reset()
        uavs, unassigned_targets = self._assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate

    def _can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        trial_sequence = uav.sequence + [target_wp]
        return self._compute_m_j(trial_sequence) >= 1

    def _assign_targets_greedily(self) -> Tuple[List[UAV], List[Waypoint]]:
        unassigned_targets = [wp for wp in self.environment.target_waypoints if wp.revenue > 0]

        while unassigned_targets:
            made_assignment = False

            for uav in self.uavs:
                if not unassigned_targets:
                    break

                best_target: Optional[Waypoint] = None
                best_rate = -1.0

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
