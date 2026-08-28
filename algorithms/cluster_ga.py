import random
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd

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
        population_size: int,
        generations: int,
        crossover_probability: float,
        mutation_probability: float,
        random_state: Optional[int] = None,
        centroids_export_path: Optional[Union[str, Path]] = None,
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
        self.centroids_export_path = (
            Path(centroids_export_path) if centroids_export_path else None
        )

        self.rng = random.Random(random_state)

    def solve(
        self,
        run_name: Optional[str] = None,
    ) -> Tuple[List[UAV], List[Waypoint], float, float]:
        """
        Solves the allocation problem using K-Means clustering + GA.
        
        Args:
            run_name: The sheet name for the current simulation run (e.g. 'Run_1', 'Sim_1').
        """
        self.reset()

        targets = [wp for wp in self.environment.target_waypoints if wp.revenue > 0]

        if not targets:
            return self.uavs, [], 0.0, 0.0

        # Step 1: Run K-means and obtain stable clusters & centroids
        clusters, stable_centroids = self._kmeans_clusters(
            targets=targets,
            k=self.num_uavs,
        )

        # Step 2: Export stable centroids to the workbook under sheet `run_name`
        if self.centroids_export_path and run_name:
            self._export_centroids_to_excel(
                centroids=stable_centroids,
                clusters=clusters,
                export_path=self.centroids_export_path,
                sheet_name=run_name,
            )

        return self.uavs, targets, 0.0, 0.0

        # Step 3: Run GA optimization per cluster
        # for uav_index, uav in enumerate(self.uavs):
        #     cluster = clusters[uav_index]

        #     if not cluster:
        #         continue

        #     best_sequence = self._ga_optimize_cluster(cluster)
        #     m_j = self._compute_m_j(best_sequence)

        #     if m_j < 1:
        #         continue

        #     uav.sequence = best_sequence
        #     uav.m_j = m_j

        # assigned_ids = {
        #     id(wp)
        #     for uav in self.uavs
        #     for wp in uav.sequence
        # }

        # unassigned_targets = [
        #     wp
        #     for wp in targets
        #     if id(wp) not in assigned_ids
        # ]

        # total_revenue = self.compute_total_revenue_all()
        # total_revenue_rate = self.compute_total_revenue_rate_all()

        # return (
        #     self.uavs,
        #     unassigned_targets,
        #     total_revenue,
        #     total_revenue_rate,
        # )

    def _export_centroids_to_excel(
        self,
        centroids: List[Tuple[float, float]],
        clusters: List[List[Waypoint]],
        export_path: Path,
        sheet_name: str,
    ) -> None:
        """
        Appends or replaces a sheet for the specific simulation run in the UAV-specific workbook.
        """
        export_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for cluster_id, (cx, cy) in enumerate(centroids):
            cluster_wps = clusters[cluster_id] if cluster_id < len(clusters) else []
            waypoint_ids = (
                [wp.wid for wp in cluster_wps]
                if cluster_wps and hasattr(cluster_wps[0], "wid")
                else []
            )

            rows.append({
                "Cluster_k": cluster_id,
                "Centroid_X": round(cx, 4),
                "Centroid_Y": round(cy, 4),
                "Assigned_Waypoints": str(waypoint_ids),
            })

        df = pd.DataFrame(rows)

        if export_path.exists():
            with pd.ExcelWriter(
                export_path,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(
                export_path,
                engine="openpyxl",
                mode="w",
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

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
    ) -> Tuple[List[List[Waypoint]], List[Tuple[float, float]]]:
        clusters: List[List[Waypoint]] = [[] for _ in range(self.num_uavs)]

        if not targets:
            return clusters, []

        active_k = min(k, len(targets))

        initial_waypoints = self.rng.sample(
            targets,
            active_k,
        )

        centroids: List[Tuple[float, float]] = [
            (float(wp.x), float(wp.y))
            for wp in initial_waypoints
        ]

        active_clusters: List[List[Waypoint]] = [[] for _ in range(active_k)]

        for iteration in range(100):
            new_clusters: List[List[Waypoint]] = [[] for _ in range(active_k)]

            for waypoint in targets:
                cluster_index = min(
                    range(active_k),
                    key=lambda index: self._squared_distance(
                        (waypoint.x, waypoint.y),
                        centroids[index],
                    ),
                )
                new_clusters[cluster_index].append(waypoint)

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

            if new_centroids == centroids:
                break

            centroids = new_centroids

        clusters[:active_k] = active_clusters
        return clusters, centroids

    def _tour_fitness(
        self,
        sequence: List[Waypoint],
    ) -> float:
        m_j = self._compute_m_j(sequence)

        if m_j < 1:
            return float("-inf")

        tour_time = self._compute_tour_flight_time(sequence, m_j)
        if tour_time <= 0.0:
            return float("-inf")

        total_revenue = m_j * self.compute_sequence_revenue(sequence)
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
        winner = max(contenders, key=self._tour_fitness)
        return winner.copy()

    def _order_crossover(
        self,
        parent_a: List[Waypoint],
        parent_b: List[Waypoint],
    ) -> List[Waypoint]:
        size = len(parent_a)
        if size < 2:
            return parent_a.copy()

        start, end = sorted(self.rng.sample(range(size), 2))
        child: List[Optional[Waypoint]] = [None for _ in range(size)]
        child[start:end + 1] = parent_a[start:end + 1]

        selected_ids = {id(wp) for wp in child if wp is not None}
        remaining_waypoints = [wp for wp in parent_b if id(wp) not in selected_ids]

        empty_positions = [index for index, waypoint in enumerate(child) if waypoint is None]
        for index, waypoint in zip(empty_positions, remaining_waypoints):
            child[index] = waypoint

        return [waypoint for waypoint in child if waypoint is not None]

    def _swap_mutation(
        self,
        sequence: List[Waypoint],
    ) -> List[Waypoint]:
        child = sequence.copy()
        if len(child) < 2:
            return child

        first_index, second_index = self.rng.sample(range(len(child)), 2)
        child[first_index], child[second_index] = (
            child[second_index],
            child[first_index],
        )
        return child

    def _ga_optimize_cluster(
        self,
        cluster: List[Waypoint],
    ) -> List[Waypoint]:
        if len(cluster) <= 1:
            return cluster.copy()

        population: List[List[Waypoint]] = [
            self.rng.sample(cluster, len(cluster))
            for _ in range(self.population_size)
        ]

        best_tour = max(population, key=self._tour_fitness).copy()
        best_fitness = self._tour_fitness(best_tour)

        for _ in range(self.generations):
            population.sort(key=self._tour_fitness, reverse=True)
            next_population: List[List[Waypoint]] = [population[0].copy()]

            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)

                if self.rng.random() < self.crossover_probability:
                    child = self._order_crossover(parent_a, parent_b)
                else:
                    child = parent_a.copy()

                if self.rng.random() < self.mutation_probability:
                    child = self._swap_mutation(child)

                next_population.append(child)

            population = next_population
            generation_best = max(population, key=self._tour_fitness)
            generation_best_fitness = self._tour_fitness(generation_best)

            if generation_best_fitness > best_fitness:
                best_tour = generation_best.copy()
                best_fitness = generation_best_fitness

        return best_tour
