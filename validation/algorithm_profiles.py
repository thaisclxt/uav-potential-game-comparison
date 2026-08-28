from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlgorithmProfile:
    """
    Describes the input/output conventions for one algorithm.

    round_column:
        Column identifying the negotiation or optimization round.

    tour_dir_name:
        Folder containing saved sequence/tour workbooks.

    output_dir_name:
        Folder where recalculated revenue-rate workbooks are saved.

    sequence_prefix:
        Prefix for UAV route columns, such as UAV0, UAV1, ...

    m_prefix:
        Prefix for repeated-tour columns, such as m_0, m_1, ...
    """
    name: str
    results_dir: Path
    waypoints_dir: Path
    round_column: str
    tour_dir_name: str = "tour"
    output_dir_name: str = "new_revenue"
    sequence_prefix: str = "UAV"
    m_prefix: str = "m_"
    m_j_mode: str = "stored"


ALGORITHM_PROFILES = {
    "IRADA": AlgorithmProfile(
        name="IRADA",
        results_dir=Path("results/IRADA"),
        waypoints_dir=Path("data/non_overlap_waypoints"),
        round_column="Round",
        m_j_mode="one",
    ),
    "greedy": AlgorithmProfile(
        name="greedy",
        results_dir=Path("results/greedy"),
        waypoints_dir=Path("data/non_overlap_waypoints"),
        round_column="negotiation_round",
        m_j_mode="stored",
    ),
    "cluster_ga": AlgorithmProfile(
        name="cluster_ga",
        results_dir=Path("results/cluster_ga"),
        waypoints_dir=Path("data/non_overlap_waypoints"),
        round_column="negotiation_round",
        m_j_mode="stored",
    ),
    "non_overlap": AlgorithmProfile(
        name="non_overlap",
        results_dir=Path("results/non_overlap"),
        waypoints_dir=Path("data/non_overlap_waypoints"),
        round_column="negotiation_round",
        m_j_mode="stored",
    ),
    "overlap": AlgorithmProfile(
        name="overlap",
        results_dir=Path("results/overlap"),
        waypoints_dir=Path("data/overlap_waypoints"),
        round_column="negotiation_round",
        m_j_mode="stored",
    ),
}