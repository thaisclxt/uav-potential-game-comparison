# Multi-UAV Task Allocation and Tour Planning

This project extends a multi-UAV task-allocation and routing simulator developed from Philip's thesis: **"Path Optimization for UAV Waypoint Navigation Using Potential Game Theory"** (Loyola Marymount University, 2025). The project evaluates the original non-overlapping and overlapping game-based approaches against two additional routing baselines:

- **Greedy allocation**
- **Cluster + Genetic Algorithm (Cluster+GA)**

The goal is to compare how different task-allocation and tour-planning strategies affect total revenue rate, per-UAV revenue contribution, and remaining flight time across different fleet sizes.

## Project background

The base simulator and experimental setting follow Philip's thesis:

- [GitHub repository link](https://github.com/Intemnets-Lab/Multi-UAV-Potential-Games/)

The project models a set of spatial waypoints and a fleet of UAVs. Each UAV receives an ordered waypoint sequence, completes a depot-to-depot flight tour, and is constrained by a maximum flight time.

The benchmark compares the original game-based methods with two added algorithms:

1. **Greedy allocator:** assigns feasible targets iteratively according to the revenue-rate objective. 
- [UAV Path Planning for Target Coverage Task in Dynamic Environment](https://ieeexplore.ieee.org/document/10130088)

2. **Cluster+GA allocator:** uses K-means to form an initial spatial grouping of tasks, then uses a genetic algorithm to improve tour order within clusters.
- [Coordinated Optimization Algorithm Combining GA with Cluster for Multi-UAVs to Multi-tasks Task Assignment and Path Planning](https://ieeexplore.ieee.org/document/8899987)


## Folder Description

| Folder or file | Purpose |
|---|---|
| `algorithms/` | Contains the waypoint-allocation algorithms and shared allocator logic. |
| `algorithms/base_allocator.py` | Defines shared functions for repeated tours, flight time, \(m_j\), revenue, and revenue rate. |
| `algorithms/greedy.py` | Implements the Greedy waypoint-allocation algorithm. |
| `algorithms/cluster_ga.py` | Implements the Cluster+GA algorithm using K-means and a genetic algorithm. |
| `src/` | Contains reusable simulation infrastructure. |
| `src/config.py` | Loads and validates configuration values from `settings.yaml`. |
| `src/environment.py` | Defines the grid environment, depot, targets, and simulation scenario. |
| `src/models.py` | Defines data models such as `UAV`, `Waypoint`, and `Depot`. |
| `src/runner.py` | Runs a selected algorithm across all requested simulations and exports results. |
| `src/io_utils.py` | Loads waypoint Excel files and exports simulation outputs to Excel. |
| `src/utils.py` | Provides utility functions such as travel-time and filename parsing. |
| `validation/` | Contains standalone helper programs for processing or validating saved results. |
| `validation/recalculate_revenue_rate.py` | Recalculates individual UAV revenue rates from stored tour sequences and \(m_j\) values. |
| `validation/assign_wp_to_cluster.py` | Produces waypoint-to-UAV cluster assignment files from Cluster+GA tour outputs. |
| `analysis/` | Contains scripts that generate figures, comparisons, and boxplots from stored simulation results. |
| `data/waypoints/` | Stores input waypoint Excel files. |
| `results/` | Stores generated Excel results, including revenue rates, tours, and cluster assignments. |
| `settings.yaml` | Stores simulation, grid, UAV, waypoint, and Cluster+GA settings. |
| `main.py` | Main entry point for running simulations. |
| `requirements.txt` | Lists third-party Python dependencies. |


## Experimental setup

Experiments are run for fleet sizes:

```text
|U| = 3, 4, 5, 6, 7, 8, 9, 10
```

For each scenario, the same waypoint instance and UAV configuration should be used across all algorithms to ensure a fair comparison.

Example common UAV settings:

```yaml
uav:
  num_uavs: <3 through 10>
  speed: 16
  max_flight_time: 1920
```

Cluster+GA configuration is stored in `settings.yaml`:

```yaml
algorithms:
  cluster_ga:
    population_size: 80
    generations: 5000
    crossover_probability: 0.60
    mutation_probability: 0.05
    random_state: 42
```

The Cluster+GA random seed controls K-means initialization and GA random operations, including population initialization, crossover selection, and mutation.

## Output structure

Simulation outputs are grouped by algorithm and UAV/grid scenario.

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
│   │   └── cluster_assignment/
│   ├── UAVs4_GRID13/
│   │   ├── revenue/
│   │   ├── tour/
│   │   └── cluster_assignment/
│   └── ...
├── greedy
│   ├── UAVs3_GRID13/
│   │   ├── revenue/
│   │   └── tour/
│   ├── UAVs4_GRID13/
│   │   ├── revenue/
│   │   └── tour/
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

1. Configure the simulation settings in `settings.yaml`.
2. Ensure the waypoint Excel files are available in `data/non_overlap_waypoints/` and `data/overlap_waypoints/`.
3. Run one algorithm at a time.
4. Review the generated Excel outputs under `results/`.
5. Run analysis scripts to generate comparison boxplots.

Run Greedy:

```bash
python main.py --algorithm greedy
```

Run Cluster+GA:

```bash
python main.py --algorithm cluster_ga
```

The two algorithms run independently and save outputs in separate directories.

## Generate comparison boxplots

Run the analysis scripts from the project root:

```bash
python -m analysis.revenue_rate_comparison
python -m analysis.per_uav_revenue_share_comparison
python -m analysis.flight_time_left_comparison
```

## Validation Scripts

Calculate revenue rates from saved tour outputs:

```bash
python -m validation.recalculate_revenue_rate
```

Create waypoint-to-UAV cluster-assignment workbooks from saved Cluster+GA sequences:

```bash
python -m validation.assign_wp_to_cluster
```

Each assignment workbook contains one sheet per simulation run and uses the final stored tour. Each sheet has the following format:

| Waypoint | UAV |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 1 |
| 3 | 2 |

An empty `UAV` value indicates that the waypoint was not assigned in the final tour.
