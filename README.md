# Multi-UAV Task Allocation and Route Planning

This project extends a multi-UAV task-allocation and routing simulator developed from Philip's thesis: **"Path Optimization for UAV Waypoint Navigation Using Potential Game Theory"** (Loyola Marymount University, 2025). The project evaluates the original non-overlapping and overlapping game-based approaches against two additional routing baselines:

- **Greedy allocation**
- **Cluster + Genetic Algorithm (Cluster+GA)**

The goal is to compare how different task-allocation and route-planning strategies affect total revenue rate, per-UAV revenue contribution, and remaining flight time across different fleet sizes.

## Project background

The base simulator and experimental setting follow Philip's thesis:

- [GitHub repository link](https://github.com/Intemnets-Lab/Multi-UAV-Potential-Games/)

The project models a set of spatial waypoints and a fleet of UAVs. Each UAV receives an ordered waypoint sequence, completes a depot-to-depot flight tour, and is constrained by a maximum flight time.

The benchmark compares the original game-based methods with two added algorithms:

1. **Greedy allocator:** assigns feasible targets iteratively according to the revenue-rate objective. 
- [UAV Path Planning for Target Coverage Task in Dynamic Environment](https://ieeexplore.ieee.org/document/10130088)

2. **Cluster+GA allocator:** uses K-means to form an initial spatial grouping of tasks, then uses a genetic algorithm to improve route order within clusters.
- [Coordinated Optimization Algorithm Combining GA with Cluster for Multi-UAVs to Multi-tasks Task Assignment and Path Planning](https://ieeexplore.ieee.org/document/8899987)

## Experimental setup

Experiments are run for fleet sizes:

```text
|U| = 3, 4, 5, 6, 7, 8, 9, 10
```

For each scenario, the same waypoint instance and UAV configuration should be used across all algorithms to ensure a fair comparison.

Recommended common settings include:

```yaml
uav:
  num_uavs: <3 through 10>
  speed: 16
  max_flight_time: 1920
```

Use fixed random seeds when reproducibility is required. For Cluster+GA, the seed controls randomized initial centroid selection and GA operations.

## Output and visualizations

The analysis scripts generate boxplots across simulation runs for each UAV count.

Output structure:

```text
results/
├── boxplots
│   ├── flight_time_left/
│   │   ├── 3uavs.png
│   │   ├── 4uavs.png
│   │   └── ...
│   ├── per_uav_revenue_share_comparisons/
│   │   ├── UAVs3/
│   │   ├── UAVs4/
│   │   └── ...
│   └── revenue_rate_comparisons/
│   │   ├── 3uavs.png
│   │   ├── 4uavs.png
│   │   └── ...
├── cluster_ga
│   ├── UAVs3_GRID13/
│   │   ├── revenue/
│   │   ├── tour/
│   ├── UAVs4_GRID13/
│   │   ├── revenue/
│   │   ├── tour/
│   └── ...
├── greedy
│   ├── UAVs3_GRID13/
│   │   ├── revenue/
│   │   ├── tour/
│   ├── UAVs4_GRID13/
│   │   ├── revenue/
│   │   ├── tour/
│   └── ...
├── irada
│   └── ...
├── non_overlap
│   └── ...
└── overlap
│   └── ...
```

## Installation

After dowloading this repository, install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
numpy
pandas
matplotlib
PyYAML
openpyxl
```

## Running experiments

1. Configure the desired number of UAVs, speed, maximum flight time, number of runs, and enabled algorithms in `settings.yaml`.
2. Run the simulation independently.
3. Run the analysis scripts to create revenue-rate, revenue-share, and flight-time-left plots.

Example commands:

```bash
python main.py --algorithm greedy
python main.py --algorithm cluster_ga

python revenue_rate_comparison.py
python per_uav_revenue_share_comparisons.py
python flight_time_left_comparison.py
```
