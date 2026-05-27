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
        ga_population_size: int = 80,
        ga_generations: int = 300,
        ga_crossover_prob: float = 0.6,
        ga_mutation_prob: float = 0.05,
        kmeans_max_iters: int = 100,
    ) -> None:
        self.environment = environment
        self.num_uavs = num_uavs
        self.uav_speed = uav_speed
        self.max_flight_time = max_flight_time

        self.ga_population_size = ga_population_size
        self.ga_generations = ga_generations
        self.ga_crossover_prob = ga_crossover_prob
        self.ga_mutation_prob = ga_mutation_prob
        self.kmeans_max_iters = kmeans_max_iters

        self.uavs: List[UAV] = [UAV(uid=i) for i in range(self.num_uavs)]

    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        """
        Cluster tasks into num_uavs groups, optimize one route per cluster with GA,
        assign one cluster to one UAV, and return the final UAV assignments,
        unassigned targets, total revenue, and total revenue rate.
        """
        self.reset()

        if not self.environment.target_waypoints:
            return self.uavs, [], 0.0, 0.0

        clusters = self._cluster_targets_kmeans(
            self.environment.target_waypoints,
            self.num_uavs,
        )

        assigned_ids = set()
        unassigned_targets: List[Waypoint] = []

        for uav, cluster in zip(self.uavs, clusters):
            if not cluster:
                uav.sequence = []
                uav.m_j = 0
                continue

            best_sequence = self._solve_cluster_tsp_ga(cluster)
            best_m_j = self._compute_m_j(best_sequence)

            if best_m_j < 1:
                unassigned_targets.extend(cluster)
                uav.sequence = []
                uav.m_j = 0
                continue

            uav.sequence = best_sequence
            uav.m_j = best_m_j

            for wp in best_sequence:
                assigned_ids.add(wp.wid)

        for wp in self.environment.target_waypoints:
            if wp.wid not in assigned_ids:
                unassigned_targets.append(wp)

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

    def compute_monitoring_frequency(self, uav: UAV) -> float:
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return (uav.m_j * self.uav_speed) / t_j

    def compute_revenue_rate(self, uav: UAV) -> float:
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)

    def compute_total_revenue_all(self) -> float:
        return sum(self.compute_total_revenue(uav) for uav in self.uavs)

    def compute_total_revenue_rate_all(self) -> float:
        return sum(self.compute_revenue_rate(uav) for uav in self.uavs)

    def _compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        """
        Compute the total flight time of a repeated tour:
        depot -> sequence repeated m_j times -> depot

        For m_j >= 1:
          total =
            depot -> first
            + m_j * (internal sequence traversal)
            + (m_j - 1) * (last -> first cycle closure)
            + last -> depot
        """
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
        """
        Compute the maximum feasible number of repetitions m_j such that
        _compute_tour_flight_time(sequence, m_j) <= max_flight_time.
        """
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
            first_repetition_time = fixed_time
            return 1 if first_repetition_time <= self.max_flight_time else 0

        cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)

        first_repetition_time = fixed_time + internal_sequence_time
        if first_repetition_time > self.max_flight_time:
            return 0

        per_extra_repetition_time = internal_sequence_time + cycle_closure_time
        remaining_time = self.max_flight_time - first_repetition_time
        extra_repetitions = int(remaining_time // per_extra_repetition_time)

        return 1 + max(extra_repetitions, 0)

    def _compute_single_cycle_time(self, sequence: List[Waypoint]) -> float:
        if not sequence:
            return 0.0

        depot = self.environment.depot
        total = travel_time(depot, sequence[0], self.uav_speed)

        for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
            total += travel_time(wp_a, wp_b, self.uav_speed)

        total += travel_time(sequence[-1], depot, self.uav_speed)
        return total

    def _cluster_targets_kmeans(
        self,
        waypoints: List[Waypoint],
        k: int,
    ) -> List[List[Waypoint]]:
        if not waypoints:
            return [[] for _ in range(k)]

        k = max(1, min(k, len(waypoints)))

        points = [(wp.x, wp.y) for wp in waypoints]
        centroids = self._initialize_centroids_farthest(points, k)

        assignments = [-1] * len(points)

        for _ in range(self.kmeans_max_iters):
            changed = False

            for i, (px, py) in enumerate(points):
                best_c = min(
                    range(k),
                    key=lambda c: (px - centroids[c][0]) ** 2 + (py - centroids[c][1]) ** 2,
                )
                if assignments[i] != best_c:
                    assignments[i] = best_c
                    changed = True

            clusters_pts = [[] for _ in range(k)]
            for idx, c in enumerate(assignments):
                clusters_pts[c].append(points[idx])

            new_centroids = []
            for c in range(k):
                if clusters_pts[c]:
                    mean_x = sum(p[0] for p in clusters_pts[c]) / len(clusters_pts[c])
                    mean_y = sum(p[1] for p in clusters_pts[c]) / len(clusters_pts[c])
                    new_centroids.append((mean_x, mean_y))
                else:
                    new_centroids.append(random.choice(points))

            centroids = new_centroids

            if not changed:
                break

        clusters = [[] for _ in range(k)]
        for idx, c in enumerate(assignments):
            clusters[c].append(waypoints[idx])

        return clusters

    def _initialize_centroids_farthest(
        self,
        points: List[Tuple[float, float]],
        k: int,
    ) -> List[Tuple[float, float]]:
        first = random.choice(points)
        centroids = [first]

        while len(centroids) < k:
            next_point = max(
                points,
                key=lambda p: min(
                    (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centroids
                ),
            )
            if next_point in centroids:
                break
            centroids.append(next_point)

        while len(centroids) < k:
            centroids.append(random.choice(points))

        return centroids

    def _route_fitness_key(self, sequence: List[Waypoint]) -> tuple[int, float]:
        m_j = self._compute_m_j(sequence)
        if m_j <= 0:
            return (0, float("inf"))
        return (-m_j, self._compute_tour_flight_time(sequence, m_j))

    def _solve_cluster_tsp_ga(self, cluster: List[Waypoint]) -> List[Waypoint]:
        if len(cluster) <= 1:
            return cluster[:]

        population = self._initialize_population(cluster)
        best = min(population, key=self._route_fitness_key)

        for _ in range(self.ga_generations):
            scored = sorted(population, key=self._route_fitness_key)
            elites = scored[: max(2, self.ga_population_size // 10)]

            if self._route_fitness_key(elites[0]) < self._route_fitness_key(best):
                best = elites[0][:]

            new_population = [route[:] for route in elites]

            while len(new_population) < self.ga_population_size:
                parent1 = self._tournament_select(scored)
                parent2 = self._tournament_select(scored)

                if random.random() < self.ga_crossover_prob:
                    child = self._ordered_crossover(parent1, parent2)
                else:
                    child = parent1[:]

                if random.random() < self.ga_mutation_prob:
                    self._swap_mutation(child)

                new_population.append(child)

            population = new_population

        return best

    def _initialize_population(self, cluster: List[Waypoint]) -> List[List[Waypoint]]:
        population = []
        base = cluster[:]

        for _ in range(self.ga_population_size):
            chrom = base[:]
            random.shuffle(chrom)
            population.append(chrom)

        return population

    def _route_distance_with_depot(self, sequence: List[Waypoint]) -> float:
        if not sequence:
            return 0.0

        depot = self.environment.depot
        total = travel_time(depot, sequence[0], self.uav_speed)

        for a, b in zip(sequence[:-1], sequence[1:]):
            total += travel_time(a, b, self.uav_speed)

        total += travel_time(sequence[-1], depot, self.uav_speed)
        return total

    def _tournament_select(
        self,
        population: List[List[Waypoint]],
        tournament_size: int = 3,
    ) -> List[Waypoint]:
        candidates = random.sample(population, min(tournament_size, len(population)))
        return min(candidates, key=self._route_fitness_key)[:]

    def _ordered_crossover(
        self,
        parent1: List[Waypoint],
        parent2: List[Waypoint],
    ) -> List[Waypoint]:
        n = len(parent1)
        if n < 2:
            return parent1[:]

        i, j = sorted(random.sample(range(n), 2))
        child: List[Optional[Waypoint]] = [None] * n

        child[i : j + 1] = parent1[i : j + 1]

        fill_values = [wp for wp in parent2 if wp not in child]
        fill_idx = 0

        for idx in range(n):
            if child[idx] is None:
                child[idx] = fill_values[fill_idx]
                fill_idx += 1

        return child

    def _swap_mutation(self, chromosome: List[Waypoint]) -> None:
        if len(chromosome) < 2:
            return
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
