import random
from typing import List, Optional, Tuple, Union

from src.environment import GridEnvironment
from src.models import Depot, UAV, Waypoint
from src.utils import travel_time


class ClusterGAAllocator:
    def __init__(
        self,
        environment: GridEnvironment,
        num_uavs: int,
        uav_speed: float,
        max_flight_time: float,
        population_size: int = 50,
        generations: int = 100,
        order_mutation_rate: float = 0.35,
        transfer_mutation_rate: float = 0.20,
        swap_uav_mutation_rate: float = 0.15,
        random_state: Optional[int] = 42,
    ) -> None:
        self.environment = environment
        self.num_uavs = num_uavs
        self.uav_speed = uav_speed
        self.max_flight_time = max_flight_time
        self.population_size = population_size
        self.generations = generations
        self.order_mutation_rate = order_mutation_rate
        self.transfer_mutation_rate = transfer_mutation_rate
        self.swap_uav_mutation_rate = swap_uav_mutation_rate
        self.rng = random.Random(random_state)

        self.uavs: List[UAV] = [UAV(uid=i) for i in range(self.num_uavs)]

    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        self.reset()
        targets = self.environment.target_waypoints.copy()

        if self.num_uavs == 0 or not targets:
            return self.uavs, targets, 0.0, 0.0

        initial_candidate = self._initial_candidate(targets)
        population = self._initial_population(initial_candidate)

        best_candidate = self._copy_candidate(initial_candidate)
        best_fitness = self._fitness(best_candidate)

        for _ in range(self.generations):
            scored_population = [
                (self._fitness(candidate), candidate)
                for candidate in population
            ]
            scored_population.sort(key=lambda item: item[0], reverse=True)

            if scored_population[0][0] > best_fitness:
                best_fitness = scored_population[0][0]
                best_candidate = self._copy_candidate(scored_population[0][1])

            elite_count = max(1, self.population_size // 10)
            next_population = [
                self._copy_candidate(candidate)
                for _, candidate in scored_population[:elite_count]
            ]

            while len(next_population) < self.population_size:
                parent = self._tournament_select(scored_population)
                next_population.append(self._mutate(parent))

            population = next_population

        self._apply_candidate(best_candidate)
        assigned_ids = {
            id(wp)
            for route in best_candidate
            for wp in route
        }
        unassigned_targets = [
            wp for wp in targets
            if id(wp) not in assigned_ids
        ]

        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return self.uavs, unassigned_targets, total_revenue, total_revenue_rate

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
        tour_time = self.current_tour_time(uav)
        if tour_time <= 0.0:
            return 0.0
        return self.compute_total_revenue(uav) / tour_time

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
        cycle_closure_time = (
            travel_time(last_wp, first_wp, self.uav_speed)
            if len(sequence) > 1
            else 0.0
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
        outbound_time = travel_time(depot, first_wp, self.uav_speed)
        return_time = travel_time(last_wp, depot, self.uav_speed)
        fixed_time = outbound_time + return_time

        if fixed_time > self.max_flight_time:
            return 0

        if len(sequence) == 1:
            round_trip = 2.0 * outbound_time
            return int(self.max_flight_time // round_trip) if round_trip > 0 else 0

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed)
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:])
        )
        cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)
        first_repetition_time = fixed_time + internal_sequence_time

        if first_repetition_time > self.max_flight_time:
            return 0

        extra_repetition_time = internal_sequence_time + cycle_closure_time
        remaining_time = self.max_flight_time - first_repetition_time
        extra_repetitions = int(remaining_time // extra_repetition_time)
        return 1 + max(extra_repetitions, 0)

    def _initial_candidate(self, targets: List[Waypoint]) -> List[List[Waypoint]]:
        candidate = self._kmeans_clusters(targets)
        repaired_candidate, _ = self._repair_candidate(candidate)
        return repaired_candidate

    def _kmeans_clusters(self, targets: List[Waypoint]) -> List[List[Waypoint]]:
        clusters: List[List[Waypoint]] = [[] for _ in range(self.num_uavs)]
        k = min(self.num_uavs, len(targets))
        if k == 0:
            return clusters

        centroids = self._kmeans_plus_plus_centroids(targets, k)

        for _ in range(100):
            new_clusters = [[] for _ in range(k)]
            for waypoint in targets:
                cluster_idx = min(
                    range(k),
                    key=lambda idx: self._squared_distance(
                        (waypoint.x, waypoint.y), centroids[idx]
                    ),
                )
                new_clusters[cluster_idx].append(waypoint)

            new_centroids = []
            for idx, cluster in enumerate(new_clusters):
                if cluster:
                    new_centroids.append((
                        sum(wp.x for wp in cluster) / len(cluster),
                        sum(wp.y for wp in cluster) / len(cluster),
                    ))
                else:
                    new_centroids.append(centroids[idx])

            clusters[:k] = new_clusters
            if new_centroids == centroids:
                break
            centroids = new_centroids

        return clusters

    def _kmeans_plus_plus_centroids(
        self,
        targets: List[Waypoint],
        k: int,
    ) -> List[Tuple[float, float]]:
        first = self.rng.choice(targets)
        centroids = [(float(first.x), float(first.y))]

        while len(centroids) < k:
            weights = [
                min(
                    self._squared_distance((wp.x, wp.y), centroid)
                    for centroid in centroids
                )
                for wp in targets
            ]
            total_weight = sum(weights)

            if total_weight <= 0.0:
                remaining = [wp for wp in targets if (wp.x, wp.y) not in centroids]
                chosen = self.rng.choice(remaining or targets)
            else:
                threshold = self.rng.random() * total_weight
                cumulative = 0.0
                chosen = targets[-1]
                for waypoint, weight in zip(targets, weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        chosen = waypoint
                        break

            centroids.append((float(chosen.x), float(chosen.y)))

        return centroids

    @staticmethod
    def _squared_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        return (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2

    def _initial_population(
        self,
        seed_candidate: List[List[Waypoint]],
    ) -> List[List[List[Waypoint]]]:
        population = [self._copy_candidate(seed_candidate)]

        while len(population) < self.population_size:
            candidate = self._copy_candidate(seed_candidate)
            for route in candidate:
                self.rng.shuffle(route)

            for _ in range(self.rng.randint(1, 4)):
                candidate = self._mutate(candidate)

            population.append(candidate)

        return population

    def _mutate(self, candidate: List[List[Waypoint]]) -> List[List[Waypoint]]:
        child = self._copy_candidate(candidate)
        roll = self.rng.random()

        if roll < self.order_mutation_rate:
            self._order_mutation(child)
        elif roll < self.order_mutation_rate + self.transfer_mutation_rate:
            self._transfer_mutation(child)
        elif roll < (
            self.order_mutation_rate
            + self.transfer_mutation_rate
            + self.swap_uav_mutation_rate
        ):
            self._swap_uav_mutation(child)
        else:
            self._order_mutation(child)

        return child if self._is_feasible(child) else self._copy_candidate(candidate)

    def _order_mutation(self, candidate: List[List[Waypoint]]) -> None:
        eligible_routes = [route for route in candidate if len(route) >= 2]
        if not eligible_routes:
            return

        route = self.rng.choice(eligible_routes)
        first_idx, second_idx = self.rng.sample(range(len(route)), 2)
        route[first_idx], route[second_idx] = route[second_idx], route[first_idx]

    def _transfer_mutation(self, candidate: List[List[Waypoint]]) -> None:
        source_indices = [idx for idx, route in enumerate(candidate) if route]
        if not source_indices or len(candidate) < 2:
            return

        source_idx = self.rng.choice(source_indices)
        destination_idx = self.rng.choice([
            idx for idx in range(len(candidate)) if idx != source_idx
        ])
        source_route = candidate[source_idx]
        destination_route = candidate[destination_idx]
        waypoint = source_route.pop(self.rng.randrange(len(source_route)))
        destination_route.insert(self.rng.randrange(len(destination_route) + 1), waypoint)

    def _swap_uav_mutation(self, candidate: List[List[Waypoint]]) -> None:
        nonempty_indices = [idx for idx, route in enumerate(candidate) if route]
        if len(nonempty_indices) < 2:
            return

        first_uav, second_uav = self.rng.sample(nonempty_indices, 2)
        first_route = candidate[first_uav]
        second_route = candidate[second_uav]
        first_idx = self.rng.randrange(len(first_route))
        second_idx = self.rng.randrange(len(second_route))
        first_route[first_idx], second_route[second_idx] = (
            second_route[second_idx],
            first_route[first_idx],
        )

    def _repair_candidate(
        self,
        candidate: List[List[Waypoint]],
    ) -> Tuple[List[List[Waypoint]], List[Waypoint]]:
        candidate = self._copy_candidate(candidate)
        deferred: List[Waypoint] = []

        for route in candidate:
            while route and self._compute_m_j(route) < 1:
                deferred.append(route.pop(self._best_removal_index(route)))

        still_unassigned: List[Waypoint] = []
        for waypoint in deferred:
            best_trial: Optional[List[List[Waypoint]]] = None
            best_fitness = float("-inf")

            for route_idx, route in enumerate(candidate):
                for insert_idx in range(len(route) + 1):
                    trial = self._copy_candidate(candidate)
                    trial[route_idx].insert(insert_idx, waypoint)
                    if not self._is_feasible(trial):
                        continue
                    trial_fitness = self._fitness(trial)
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_trial = trial

            if best_trial is None:
                still_unassigned.append(waypoint)
            else:
                candidate = best_trial

        return candidate, still_unassigned

    def _best_removal_index(self, route: List[Waypoint]) -> int:
        best_idx = 0
        best_score = float("-inf")

        for idx in range(len(route)):
            trial_route = route[:idx] + route[idx + 1:]
            if not trial_route:
                return idx

            m_j = self._compute_m_j(trial_route)
            if m_j < 1:
                score = float("-inf")
            else:
                tour_time = self._compute_tour_flight_time(trial_route, m_j)
                score = m_j * self.compute_sequence_revenue(trial_route) / tour_time

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _is_feasible(self, candidate: List[List[Waypoint]]) -> bool:
        seen_ids = set()

        for route in candidate:
            if not route:
                continue
            if self._compute_m_j(route) < 1:
                return False

            for waypoint in route:
                waypoint_id = id(waypoint)
                if waypoint_id in seen_ids:
                    return False
                seen_ids.add(waypoint_id)

        return True

    def _fitness(self, candidate: List[List[Waypoint]]) -> float:
        if not self._is_feasible(candidate):
            return float("-inf")

        total_rate = 0.0
        for route in candidate:
            if not route:
                continue

            m_j = self._compute_m_j(route)
            tour_time = self._compute_tour_flight_time(route, m_j)
            total_revenue = m_j * self.compute_sequence_revenue(route)
            total_rate += total_revenue / tour_time

        return total_rate

    def _tournament_select(
        self,
        scored_population: List[Tuple[float, List[List[Waypoint]]]],
        tournament_size: int = 3,
    ) -> List[List[Waypoint]]:
        contenders = self.rng.sample(
            scored_population,
            k=min(tournament_size, len(scored_population)),
        )
        _, winner = max(contenders, key=lambda item: item[0])
        return self._copy_candidate(winner)

    @staticmethod
    def _copy_candidate(
        candidate: List[List[Waypoint]],
    ) -> List[List[Waypoint]]:
        return [route.copy() for route in candidate]

    def _apply_candidate(self, candidate: List[List[Waypoint]]) -> None:
        self.reset()
        for uav, route in zip(self.uavs, candidate):
            uav.sequence = route.copy()
            uav.m_j = self._compute_m_j(uav.sequence)