import math, re

from pathlib import Path
from .models import Depot, Waypoint


def euclidean_distance(a: Waypoint | Depot, b: Waypoint | Depot) -> float:
    """
    Calculate the Euclidean distance between two points (waypoints or depot).
    """
    return math.hypot(b.x - a.x, b.y - a.y)


def travel_time(a: Waypoint | Depot, b: Waypoint | Depot, uav_speed: float) -> float:
    """
    Calculate the travel time between two points (waypoints or depot) based on the UAV speed.
    """
    return euclidean_distance(a, b) / uav_speed


def extract_num_uavs(file_path: str) -> int:
    """
    Extract the number of UAVs (m) from the filename using a regular expression.
    (e.g., from UAVs10_GRID13_waypoints.xlsx it will extract 10)
    """
    match = re.search(r"UAVs(\d+)", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not infer number of UAVs from filename: {file_path}")
    return int(match.group(1))


def extract_grid_size(file_path: str) -> int:
    """
    Extract the grid size from the filename using a regular expression.
    (e.g., from UAVs10_GRID13_waypoints.xlsx it will extract 13)
    """
    match = re.search(r"GRID(\d+)", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not infer grid size from filename: {file_path}")
    return int(match.group(1))
