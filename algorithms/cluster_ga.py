import random
from typing import List, Optional, Tuple

from algorithms.base_allocator import BaseAllocator
from src.environment import GridEnvironment
from src.models import UAV, Waypoint


class ClusterGAAllocator(BaseAllocator):
    def __init__(
        self,
        environment: GridEnvironment,
        num_uavs: int,
        uav_speed: float,
        max_flight_time: float,
        population_size: int = 80,
        generations: int = 5000,
        crossover_probability: float = 0.60,
        mutation_probability: float = 0.05,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            environment=environment,
            num_uavs=num_uavs,
            uav_speed=uav_speed,
            max_flight_time=max_flight_time,
        )

        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

        self.rng = random.Random(random_state)

    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        self.reset()

        targets = [ wp for wp in self.environment.target_waypoints if wp.revenue > 0]

        if not targets:
            return self.uavs, [], 0.0, 0.0

        clusters = self._kmeans_clusters(
            targets=targets,
            k=self.num_uavs,
        )

        for uav_index, uav in enumerate(self.uavs):
            cluster = clusters[uav_index]

            if not cluster:
                continue

            best_sequence = self._ga_optimize_cluster(cluster)
            m_j = self._compute_m_j(best_sequence)

            # Fixed cluster remains unassigned if no feasible tour exists.
            if m_j < 1:
                continue

            uav.sequence = best_sequence
            uav.m_j = m_j

        assigned_ids = {
            id(wp)
            for uav in self.uavs
            for wp in uav.sequence
        }

        unassigned_targets = [
            wp
            for wp in targets
            if id(wp) not in assigned_ids
        ]

        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()

        return (
            self.uavs,
            unassigned_targets,
            total_revenue,
            total_revenue_rate,
        )

    # ============================================================
    # K-means: fixed clusters, k = num_uavs
    # ============================================================

    @staticmethod
    def _squared_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        return (
            (point_a[0] - point_b[0]) ** 2
            + (point_a[1] - point_b[1]) ** 2
        )

    def _kmeans_clusters(
        self,
        targets: List[Waypoint],
        k: int,
    ) -> List[List[Waypoint]]:
        """
        Standard K-means.

        Initial centroids are distinct target locations chosen randomly.
        This is closer to the paper than K-means++ initialization.
        """
        clusters: List[List[Waypoint]] = [
            []
            for _ in range(self.num_uavs)
        ]

        if not targets:
            return clusters

        active_k = min(k, len(targets))

        initial_waypoints = self.rng.sample(
            targets,
            active_k,
        )

        centroids: List[Tuple[float, float]] = [
            (float(wp.x), float(wp.y))
            for wp in initial_waypoints
        ]

        active_clusters: List[List[Waypoint]] = [
            []
            for _ in range(active_k)
        ]

        for iteration in range(100):
            new_clusters: List[List[Waypoint]] = [
                []
                for _ in range(active_k)
            ]

            # Assign every waypoint to nearest centroid.
            for waypoint in targets:
                cluster_index = min(
                    range(active_k),
                    key=lambda index: self._squared_distance(
                        (waypoint.x, waypoint.y),
                        centroids[index],
                    ),
                )

                new_clusters[cluster_index].append(waypoint)

            # Recalculate centroid as mean target location.
            new_centroids: List[Tuple[float, float]] = []

            for index, cluster in enumerate(new_clusters):
                if not cluster:
                    new_centroids.append(centroids[index])
                    continue

                new_centroids.append(
                    (
                        sum(wp.x for wp in cluster) / len(cluster),
                        sum(wp.y for wp in cluster) / len(cluster),
                    )
                )

            active_clusters = new_clusters

            # Stable centroids: stop K-means and begin GA.
            if new_centroids == centroids:
                print(
                    f"[ClusterGA][KMeans] Converged after "
                    f"{iteration} iteration(s)."
                )
                break

            centroids = new_centroids

        clusters[:active_k] = active_clusters

        return clusters

    def _tour_fitness(
        self,
        sequence: List[Waypoint],
    ) -> float:
        m_j = self._compute_m_j(sequence)

        if m_j < 1:
            return float("-inf")

        tour_time = self._compute_tour_flight_time(
            sequence,
            m_j,
        )

        if tour_time <= 0.0:
            return float("-inf")

        total_revenue = (
            m_j
            * self.compute_sequence_revenue(sequence)
        )

        return total_revenue / tour_time

    def _tournament_select(
        self,
        population: List[List[Waypoint]],
        tournament_size: int = 3,
    ) -> List[Waypoint]:
        contenders = self.rng.sample(
            population,
            k=min(tournament_size, len(population)),
        )

        winner = max(
            contenders,
            key=self._tour_fitness,
        )

        return winner.copy()

    def _order_crossover(
        self,
        parent_a: List[Waypoint],
        parent_b: List[Waypoint],
    ) -> List[Waypoint]:
        """
        Order crossover.

        The child contains every waypoint in the fixed cluster exactly once.
        """
        size = len(parent_a)

        if size < 2:
            return parent_a.copy()

        start, end = sorted(
            self.rng.sample(range(size), 2)
        )

        child: List[Optional[Waypoint]] = [
            None
            for _ in range(size)
        ]

        child[start:end + 1] = parent_a[start:end + 1]

        selected_ids = {
            id(wp)
            for wp in child
            if wp is not None
        }

        remaining_waypoints = [
            wp
            for wp in parent_b
            if id(wp) not in selected_ids
        ]

        empty_positions = [
            index
            for index, waypoint in enumerate(child)
            if waypoint is None
        ]

        for index, waypoint in zip(
            empty_positions,
            remaining_waypoints,
        ):
            child[index] = waypoint

        return [
            waypoint
            for waypoint in child
            if waypoint is not None
        ]

    def _swap_mutation(
        self,
        sequence: List[Waypoint],
    ) -> List[Waypoint]:
        """Swap two waypoint positions in the same UAV tour."""
        child = sequence.copy()

        if len(child) < 2:
            return child

        first_index, second_index = self.rng.sample(
            range(len(child)),
            2,
        )

        child[first_index], child[second_index] = (
            child[second_index],
            child[first_index],
        )

        return child

    def _ga_optimize_cluster(
        self,
        cluster: List[Waypoint],
    ) -> List[Waypoint]:
        """
        Run an independent GA for one fixed K-means cluster.

        Chromosome:
            A permutation of the cluster's waypoints.

        Operators:
            Tournament selection
            Order crossover with probability 0.60
            Swap mutation with probability 0.05
            Elitism: retain best tour each generation
        """
        if len(cluster) <= 1:
            return cluster.copy()

        population: List[List[Waypoint]] = [
            self.rng.sample(cluster, len(cluster))
            for _ in range(self.population_size)
        ]

        best_tour = max(
            population,
            key=self._tour_fitness,
        ).copy()

        best_fitness = self._tour_fitness(best_tour)

        for _ in range(self.generations):
            population.sort(
                key=self._tour_fitness,
                reverse=True,
            )

            next_population: List[List[Waypoint]] = [
                population[0].copy()
            ]

            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)

                if self.rng.random() < self.crossover_probability:
                    child = self._order_crossover(
                        parent_a,
                        parent_b,
                    )
                else:
                    child = parent_a.copy()

                if self.rng.random() < self.mutation_probability:
                    child = self._swap_mutation(child)

                next_population.append(child)

            population = next_population

            generation_best = max(
                population,
                key=self._tour_fitness,
            )

            generation_best_fitness = self._tour_fitness(
                generation_best
            )

            if generation_best_fitness > best_fitness:
                best_tour = generation_best.copy()
                best_fitness = generation_best_fitness

        return best_tour
