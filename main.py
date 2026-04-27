import math, random, os, re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union
from matplotlib.animation import PillowWriter

# ============================================================
# Configuration
# 
# Note: USE_EXTERNAL_WAYPOINTS = True means the simulation will load waypoints from generated Excel files in the WAYPOINTS_ROOT directory.
# If you want to run the simulation with a default grid environment set USE_EXTERNAL_WAYPOINTS = False and adjust the grid parameters as needed.
# ============================================================

USE_EXTERNAL_WAYPOINTS = True
DEPOT_LOCATION = (0, 0)

OUTPUT_BASE_DIR = "Results"                 # Base directory for saving results (Excel files)
WAYPOINTS_PATH = Path("waypoints")          # Directory containing the generated Excel files with waypoints (used when USE_EXTERNAL_WAYPOINTS is True)
SETTINGS_PATH = Path("settings.yaml")       # Path to the YAML file containing configuration parameters for the simulation

# ============================================================
# Data models
# ============================================================

@dataclass
class SimpleConfig:
    results_dir: str
    visualization_dir: str
    speed: float
    max_flight_time: float

def load_simple_config(config: dict) -> SimpleConfig:
    project = config.get("project", {})
    uav = config.get("uav", {})
    return SimpleConfig(
        results_dir=project.get("results_dir", "Results"),
        visualization_dir=project.get("visualization_dir", "Visualizations"),
        speed=float(uav.get("speed", 16)),
        max_flight_time=float(uav.get("max_flight_time", 1920)),
    )

def _algo_label_from_seq_file(seq_file: Path) -> Optional[str]:
    stem = seq_file.stem.replace("_sequences", "")
    if "IRADA" in stem:
        return "IRADA"
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.startswith("Mode"):
            if i + 1 < len(parts):
                return f"{p}_{parts[i + 1]}"
            return p
    return None


def _mode_from_path(p: Path) -> str:
    parts = p.parts
    if "NonOverlap" in parts:
        return "NonOverlap"
    if "Overlap" in parts:
        return "Overlap"
    if "IRADA" in parts or "Benchmarking" in parts:
        return "IRADA"
    return "Other"


def vis_gifs_root_for_sim(vis_root: Path, mode: str, seq_sim_path: Path) -> Path:
    date = seq_sim_path.parent.name  # YYYY-MM-DD
    sim_name = seq_sim_path.name     # simulation_k
    return vis_root / mode / "gifs" / date / sim_name


def get_revenue_file_for_sequence(seq_file: Path, rev_dir: Path) -> Path | None:
    stem = seq_file.stem.replace("_sequences", "")
    parts = stem.split("_")

    if "IRADA" in parts:
        rev_stem = f"{parts[0]}_{parts[1]}_IRADA"
        cand = rev_dir / f"{rev_stem}.xlsx"
        if cand.exists():
            return cand
        cand2 = rev_dir / f"{stem}.xlsx"
        if cand2.exists():
            return cand2
        return None

    if parts[-1] == "Greedy":
        cand = rev_dir / f"{parts[0]}_{parts[1]}_Greedy.xlsx"
        if cand.exists():
            return cand

    mode_idx = next((i for i, p in enumerate(parts) if p.startswith("Mode")), None)
    if mode_idx is not None and mode_idx + 1 < len(parts):
        rev_stem = f"{parts[0]}_{parts[1]}_{parts[mode_idx]}_{parts[mode_idx + 1]}"
        cand = rev_dir / f"{rev_stem}.xlsx"
        if cand.exists():
            return cand

    cand2 = rev_dir / f"{stem}.xlsx"
    if cand2.exists():
        return cand2

    return None

def generate_blockspot_gifs(
    results_base_dir: str,
    waypoints_root: Path,
    cfg: SimpleConfig,
) -> None:
    """
    Generate GIFs with grid blockspots for the latest Greedy simulation (today's date):
    - reads sequences from results/sequences/YYYY-MM-DD/simulation_1
    - reads revenue from   results/revenue/YYYY-MM-DD/simulation_1
    - finds waypoints by UAVsM_GRIDN_waypoints.xlsx anywhere under waypoints_root
    Saves GIFs to Visualizations/Greedy/gifs/YYYY-MM-DD/simulation_1/<algo>/*.gif
    """
    here = Path(__file__).parent

    # Use today's date for results
    date_str = datetime.now().strftime("%Y-%m-%d")
    seq_sim_dir = Path(results_base_dir) / "sequences" / date_str / "simulation_1"
    rev_sim_dir = Path(results_base_dir) / "revenue"   / date_str / "simulation_1"

    if not seq_sim_dir.exists():
        print(f"[GIF] sequences dir not found: {seq_sim_dir}")
        return
    if not rev_sim_dir.exists():
        print(f"[GIF] revenue dir not found: {rev_sim_dir}")
        return

    mode = "Greedy"
    vis_root = here / cfg.visualization_dir
    gifs_sim_root = vis_gifs_root_for_sim(vis_root, mode, seq_sim_dir)
    gifs_sim_root.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zero_color = "#CCCCCC"
    nonzero_color = "#111111"

    # Loop over each *_sequences.xlsx in this simulation
    for seq_file in seq_sim_dir.glob("*_sequences.xlsx"):
        base_stem = seq_file.stem.replace("_sequences", "")
        parts = base_stem.split("_")

        algo_label = _algo_label_from_seq_file(seq_file) or "Greedy"

        rev_dir = rev_sim_dir

        rev_f = get_revenue_file_for_sequence(seq_file, rev_dir)
        if rev_f is None or not rev_f.exists():
            print(f"[GIF] Skipping {base_stem}: revenue file not found.")
            continue

        # Waypoints filename pattern: UAVsM_GRIDN_waypoints.xlsx
        if len(parts) < 2:
            print(f"[GIF] Skipping {base_stem}: cannot parse UAVs/GRID.")
            continue

        wp_pattern = f"{parts[0]}_{parts[1]}_waypoints.xlsx"
        wp_matches = sorted(waypoints_root.rglob(wp_pattern))
        if not wp_matches:
            print(f"[GIF] Skipping {base_stem}: no waypoint file found matching {wp_pattern}")
            continue

        wp_f = wp_matches[0]

        rev_sheets = pd.read_excel(rev_f, sheet_name=None, index_col=0)
        seq_sheets = pd.read_excel(seq_file, sheet_name=None, index_col=0)

        # Get UAV count and grid size from filename tokens
        try:
            n_uavs = int(parts[0][4:])
            grid_dim = int(parts[1][4:])
        except Exception:
            n_uavs = 0
            grid_dim = 0

        cfg_gifs_dir = gifs_sim_root / algo_label
        cfg_gifs_dir.mkdir(parents=True, exist_ok=True)

        for run_name, seq_df in seq_sheets.items():
            rev_df = rev_sheets.get(run_name)
            if rev_df is None:
                continue

            # Align revenue length to sequences (forward-fill)
            n_seq = len(seq_df)
            n_rev = len(rev_df)
            if n_rev < n_seq:
                last_row = rev_df.iloc[-1]
                extra = pd.DataFrame([last_row] * (n_seq - n_rev), columns=rev_df.columns)
                rev_df = pd.concat([rev_df, extra], ignore_index=True)
                print(
                    f"[GIF] Extended revenue {base_stem}:{run_name} "
                    f"from {n_rev} to {n_seq} rows."
                )
            elif n_rev > n_seq:
                rev_df = rev_df.iloc[:n_seq].copy()
                print(
                    f"[GIF] Truncated revenue {base_stem}:{run_name} "
                    f"from {n_rev} to {n_seq} rows."
                )

            seq_df = seq_df.reset_index(drop=True)
            rev_df = rev_df.reset_index(drop=True)

            df_wp = pd.read_excel(wp_f, sheet_name=run_name)
            coords = {
                int(r.Waypoint): (float(r.X), float(r.Y), float(r.Revenue))
                for _, r in df_wp.iterrows()
            }

            xs_sorted = [coords[i][0] for i in sorted(coords)]
            if len(xs_sorted) >= 2:
                d = abs(xs_sorted[1] - xs_sorted[0])
            else:
                d = 1.0

            xs_all = [c[0] for c in coords.values()]
            ys_all = [c[1] for c in coords.values()]
            x_span = max(xs_all) - min(xs_all)
            y_span = max(ys_all) - min(ys_all)

            ips = 0.5  # inches per spacing
            sidebar = 2.5
            fig_w = x_span * ips + sidebar
            fig_h = y_span * ips + 1.0

            fig = plt.figure(figsize=(fig_w, fig_h))

            header = (
                f"Greedy UAVs = {n_uavs} Grid = {grid_dim} × {grid_dim}\n"
                f"Simulation Run = {run_name.replace('SimRun', '')}"
            )
            fig.text(0.5, 0.98, header, ha="center", va="top", fontsize=10)

            ax = fig.add_axes([0.05, 0.1, 0.7, 0.8])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal", adjustable="box")

            wp_rects = {}
            for wid, (x, y, rev_val) in coords.items():
                rect = plt.Rectangle(
                    (x - 0.5 * d, y - 0.5 * d),
                    d,
                    d,
                    facecolor=zero_color if rev_val <= 0 else nonzero_color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                wp_rects[wid] = rect

            ax.set_xlim(min(xs_all) - d, max(xs_all) + d)
            ax.set_ylim(min(ys_all) - d, max(ys_all) + d)

            uav_cols = [c for c in seq_df.columns if str(c).upper().startswith("UAV")]
            uav_colors = {j: colors[j % len(colors)] for j in range(len(uav_cols))}

            paths = {
                j: ax.plot([], [], color=uav_colors[j], linewidth=1.0)[0]
                for j in range(len(uav_cols))
            }

            def frame_to_paths(frame: int):
                for j, ucol in enumerate(uav_cols):
                    seq_str = str(seq_df.iloc[frame][ucol])
                    if not seq_str or seq_str.lower() == "nan":
                        paths[j].set_data([], [])
                        continue
                    ids = [int(x) for x in seq_str.split("-") if x]
                    xs = [coords[i][0] for i in ids if i in coords]
                    ys = [coords[i][1] for i in ids if i in coords]
                    paths[j].set_data(xs, ys)

            time_text = fig.text(0.78, 0.9, "", ha="left", va="center", fontsize=10)
            rev_text = fig.text(0.78, 0.85, "", ha="left", va="center", fontsize=10)

            def update(frame: int):
                frame_to_paths(frame)
                t = frame * cfg.max_flight_time / max(1, len(seq_df) - 1)
                time_text.set_text(f"Round {frame}\nTime ≈ {t:.1f}s")
                tot = 0.0
                for c in rev_df.columns:
                    if str(c).upper().startswith("UAV"):
                        tot += float(rev_df.iloc[frame][c])
                rev_text.set_text(f"Total revenue rate:\n{tot:.1f}")
                return list(paths.values()) + [time_text, rev_text]

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(seq_df),
                interval=300,
                blit=False,
                repeat=False,
            )

            gif_name = f"{base_stem}_{run_name}.gif"
            out_path = cfg_gifs_dir / gif_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[GIF] Saving {out_path}")
            writer = PillowWriter(fps=3)
            anim.save(str(out_path), writer=writer)
            plt.close(fig)

@dataclass
class Waypoint:
    """
    Represents a target waypoint with coordinates, unique ID, and revenue.
    """
    x: float
    y: float
    wid: int
    revenue: float


@dataclass
class Depot:
    """
    Represents the depot location.
    """
    x: float
    y: float


@dataclass
class UAV:
    """
    Represents a UAV with a unique ID, assigned sequence of waypoints, and number of repetitions.
    """
    uid: int
    sequence: List[Waypoint] = field(default_factory=list)  # S_j
    m_j: int = 0                                            # number of repetitions

    def reset(self) -> None:
        """
        Reset the UAV's assigned sequence and repetitions for a new simulation run.
        """
        self.sequence.clear()
        self.m_j = 0


# ============================================================
# Environment
# ============================================================

class GridEnvironment:
    """
    Represents the grid environment with a depot and a set of target waypoints.
    Can be provided externally from generated Excel files or built as a default grid.
    """
    def __init__(
        self,
        *,
        external_targets: Optional[List[Waypoint]] = None,
        depot_location: Tuple[float, float],
        base_revenue: float,
        base_seed: int,
        use_fixed_targets: bool,
        fixed_target_indices: List[int],
        grid_width: int,
        grid_height: int,
        grid_spacing: float,
        min_wp_revenue: float,
        max_wp_revenue: float,
        num_targets: int,
    ):
        self.depot: Depot = Depot(x=depot_location[0], y=depot_location[1])
        self.base_revenue = base_revenue
        self.min_wp_revenue = min_wp_revenue
        self.max_wp_revenue = max_wp_revenue
        self.use_fixed_targets = use_fixed_targets
        self.fixed_target_indices = fixed_target_indices
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_spacing = grid_spacing
        self.num_targets = num_targets

        random.seed(base_seed)

        if external_targets is not None:
            self.waypoints: List[Waypoint] = external_targets.copy()
            self.target_waypoints: List[Waypoint] = [wp for wp in self.waypoints if wp.revenue > 0]
        else:
            self.waypoints: List[Waypoint] = self._build_grid()
            self.target_waypoints: List[Waypoint] = self._select_waypoint_targets()

    def _build_grid(self) -> List[Waypoint]:
        """
        Build a grid of waypoints based on the specified GRID_WIDTH, GRID_HEIGHT, and GRID_SPACING.
        """
        waypoints: List[Waypoint] = []
        wid = 0
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if (i, j) == (self.depot.x, self.depot.y):
                    continue
                x = j * self.grid_spacing
                y = i * self.grid_spacing
                waypoints.append(Waypoint(x=x, y=y, wid=wid, revenue=self.base_revenue))
                wid += 1
        return waypoints
    
    def _select_waypoint_targets(self) -> List[Waypoint]:
        """
        Select a fixed subset of waypoints to be the target waypoints.
        """
        if self.use_fixed_targets:
            return [self.waypoints[i] for i in self.fixed_target_indices]
        return random.sample(self.waypoints, self.num_targets)

    def assign_random_revenues(self) -> None:
        for wp in self.target_waypoints:
            wp.revenue = random.uniform(self.min_wp_revenue, self.max_wp_revenue)

    def print_summary(self) -> None:
        """
        Print a summary of the environment, including depot location, waypoints, and target waypoints.
        """
        print(f"Depot at: ({self.depot.x}, {self.depot.y})\n")

        for wp in sorted(self.waypoints, key=lambda w: w.wid):
            print(f"Waypoint {wp.wid}: ({wp.x}, {wp.y})")

        print(f"\nTotal waypoints: {len(self.waypoints)}")
        print(f"Number of targets: {len(self.target_waypoints)}\n")

        for wp in sorted(self.target_waypoints, key=lambda w: w.wid):
            print(f"Waypoint target {wp.wid}: ({wp.x}, {wp.y}) with revenue {wp.revenue:.2f}")


# ============================================================
# Utility functions
# ============================================================

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


def load_waypoints_sheet(waypoints_file: str, sheet_name: str) -> List[Waypoint]:
    """
    Load waypoints from a specific sheet in the given Excel file. The sheet must contain columns:
    - "Waypoint": unique ID of the waypoint
    - "Revenue": revenue value for the waypoint
    - "X": x-coordinate of the waypoint
    - "Y": y-coordinate of the waypoint
    """
    df = pd.read_excel(waypoints_file, sheet_name=sheet_name)
    required_cols = {"Waypoint", "Revenue", "X", "Y"}
    
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Sheet {sheet_name} in {waypoints_file} must contain columns {required_cols}, "
            f"but has {list(df.columns)}"
        )

    targets: List[Waypoint] = []
    for _, row in df.iterrows():
        targets.append(
            Waypoint(
                x=float(row["X"]),
                y=float(row["Y"]),
                wid=int(row["Waypoint"]),
                revenue=float(row["Revenue"]),
            )
        )

    return targets


# ============================================================
# Greedy allocator
# ============================================================

class GreedyAllocator:
    """
    Implements a greedy algorithm to assign target waypoints to UAVs while maximizing revenue rate.
    """
    def __init__(        
            self,
            environment: GridEnvironment,
            num_uavs: int,
            uav_speed: float,
            max_flight_time: float,
        ):
        self.environment = environment
        self.num_uavs = num_uavs
        self.uav_speed = uav_speed
        self.max_flight_time = max_flight_time

        self.uavs = [UAV(uid=i) for i in range(num_uavs)]

    # ---------- Time and tour helpers ----------

    def compute_tour_flight_time(self, sequence: List[Waypoint], m_j: int) -> float:
        """
        Compute the total flight time of a tour given a sequence of waypoints and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not sequence or m_j <= 0:
            return 0.0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        total = travel_time(depot, first_wp, self.uav_speed)

        for rep in range(m_j):
            for wp_a, wp_b in zip(sequence[:-1], sequence[1:]):
                total += travel_time(wp_a, wp_b, self.uav_speed)

            if rep < m_j - 1:
                total += travel_time(last_wp, first_wp, self.uav_speed)

        total += travel_time(last_wp, depot, self.uav_speed)
        return total

    def compute_m_j(self, sequence: List[Waypoint]) -> int:
        """
        Compute the maximum number of repetitions (m_j) for a given sequence of waypoints such that the total tour time does not exceed MAX_FLIGHT_TIME.
        """
        if not sequence:
            return 0

        depot = self.environment.depot
        first_wp = sequence[0]
        last_wp = sequence[-1]

        depot_legs_time = (
            travel_time(depot, first_wp, self.uav_speed)
            + travel_time(last_wp, depot, self.uav_speed)
        )

        if depot_legs_time > self.max_flight_time:
            return 0

        internal_sequence_time = sum(
            travel_time(wp_a, wp_b, self.uav_speed) for wp_a, wp_b in zip(sequence[:-1], sequence[1:])
        )

        if len(sequence) == 1:
            return 1
        
        cycle_closure_time = travel_time(last_wp, first_wp, self.uav_speed)
        repeated_cycle_time = internal_sequence_time + cycle_closure_time

        if repeated_cycle_time <= 0:
            return 1

        remaining_time = self.max_flight_time - depot_legs_time + cycle_closure_time
        max_repetitions = int(remaining_time // repeated_cycle_time)
        return max(max_repetitions, 1)

    def build_tour(self, uav: UAV) -> List[Union[Depot, Waypoint]]:
        """
        Build the full tour for a UAV based on its assigned sequence and number of repetitions.
        The tour starts and ends at the depot, and visits the waypoints in the sequence m_j times.
        """
        if not uav.sequence or uav.m_j <= 0:
            return [self.environment.depot]
        return [self.environment.depot] + (uav.sequence * uav.m_j) + [self.environment.depot]

    def current_tour_time(self, uav: UAV) -> float:
        """
        Compute the current tour flight time for a UAV based on its assigned sequence and number of repetitions.
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return self.compute_tour_flight_time(uav.sequence, uav.m_j)

    # ---------- Revenue and revenue-rate ----------

    def compute_sequence_revenue(self, sequence: List[Waypoint]) -> float:
        """
        Compute the total revenue of a sequence of waypoints by summing their individual revenues.
        """
        return sum(wp.revenue for wp in sequence)

    def compute_total_revenue(self, uav: UAV) -> float:
        """
        Compute the total revenue for a UAV based on its assigned sequence and number of repetitions.
        The total revenue is the revenue of the sequence multiplied by the number of repetitions (m_j).
        """
        if not uav.sequence or uav.m_j <= 0:
            return 0.0
        return uav.m_j * self.compute_sequence_revenue(uav.sequence)

    def compute_monitoring_frequency(self, uav: UAV) -> float:
        """
        Compute the monitoring frequency for a UAV based on its assigned sequence and number of repetitions.
        """
        t_j = self.current_tour_time(uav)
        if t_j <= 0.0:
            return 0.0
        return (uav.m_j * self.uav_speed) / t_j

    def compute_revenue_rate(self, uav: UAV) -> float:
        """
        Compute the revenue rate for a UAV as the product of its monitoring frequency and total revenue.
        """
        return self.compute_monitoring_frequency(uav) * self.compute_total_revenue(uav)

    # ---------- Greedy assignment logic ----------

    def reset(self) -> None:
        """
        Reset all UAVs to their initial state (empty sequence and m_j = 0) for a new simulation run.
        """
        for uav in self.uavs:
            uav.reset()

    def can_assign_target(self, uav: UAV, target_wp: Waypoint) -> bool:
        """
        Check if assigning the target waypoint to the UAV's sequence would still allow for a feasible tour (m_j >= 1).
        """
        trial_sequence = uav.sequence + [target_wp]
        trial_m_j = self.compute_m_j(trial_sequence)
        return trial_m_j >= 1

    def find_nearest_feasible_target(self, uav: UAV, unassigned_targets: List[Waypoint]) -> Optional[Waypoint]:
        """
        Find the nearest unassigned target waypoint that can be feasibly assigned to the UAV's sequence without exceeding MAX_FLIGHT_TIME.
        """
        if not unassigned_targets:
            return None

        current_location = uav.sequence[-1] if uav.sequence else self.environment.depot
        feasible_targets = [wp for wp in unassigned_targets if self.can_assign_target(uav, wp)]

        if not feasible_targets:
            return None

        min_distance = min(euclidean_distance(current_location, wp) for wp in feasible_targets)
        candidates = [wp for wp in feasible_targets if euclidean_distance(current_location, wp) == min_distance]
        return random.choice(candidates)

    def select_uav_with_min_tour_time(self) -> UAV:
        """
        Select the UAV with the minimum current tour time. If multiple UAVs have the same minimum tour time, select one randomly among them.
        """
        min_time = min(self.current_tour_time(uav) for uav in self.uavs)
        candidates = [uav for uav in self.uavs if self.current_tour_time(uav) == min_time]
        return random.choice(candidates)

    def assign_targets_greedily(self) -> Tuple[List[UAV], List[Waypoint]]:
        """
        Greedily assign target waypoints to UAVs by repeatedly selecting the UAV with the minimum current tour time and assigning it the nearest feasible target waypoint until no more assignments are possible.
        """
        unassigned_targets = self.environment.target_waypoints.copy()

        while unassigned_targets:
            selected_uav = self.select_uav_with_min_tour_time()
            # print(f"Selected UAV {selected_uav.uid} with current tour time {self.current_tour_time(selected_uav):.2f} seconds for assignment.")
            nearest_target = self.find_nearest_feasible_target(selected_uav, unassigned_targets)

            if nearest_target is None:
                break

            selected_uav.sequence.append(nearest_target)
            selected_uav.m_j = self.compute_m_j(selected_uav.sequence)
            unassigned_targets.remove(nearest_target)

        return self.uavs, unassigned_targets

    # ---------- Aggregate metrics ----------

    def compute_total_revenue_all(self) -> float:
        """
        Compute the total revenue across all UAVs by summing their individual total revenues.
        """
        return sum(self.compute_total_revenue(uav) for uav in self.uavs)

    def compute_total_revenue_rate_all(self) -> float:
        """
        Compute the total revenue rate across all UAVs by summing their individual revenue rates.
        """
        return sum(self.compute_revenue_rate(uav) for uav in self.uavs)

    def solve(self) -> Tuple[List[UAV], List[Waypoint], float, float]:
        """
        Run the greedy assignment algorithm and return the final UAV assignments, unassigned targets, total revenue, and total revenue rate.
        """
        self.reset()
        uavs, unassigned_targets = self.assign_targets_greedily()
        total_revenue = self.compute_total_revenue_all()
        total_revenue_rate = self.compute_total_revenue_rate_all()
        return uavs, unassigned_targets, total_revenue, total_revenue_rate


# ============================================================
# Output helpers
# ============================================================

def print_solution(
    environment: GridEnvironment,
    allocator: GreedyAllocator,
    uavs: List[UAV],
    unassigned_targets: List[Waypoint],
    total_revenue: float,
    total_revenue_rate: float,
    header: str = "Greedy Result",
) -> None:
    print(f"\n=== {header} ===")
    print(f"Number of UAVs (m): {allocator.num_uavs}")
    print(f"Selected targets: {sorted([wp.wid for wp in environment.target_waypoints])}")

    for uav in uavs:
        seq = uav.sequence
        tour = allocator.build_tour(uav)
        tour_time = allocator.current_tour_time(uav)
        remaining_time = allocator.max_flight_time - tour_time

        print(f"\nUAV {uav.uid}")
        print(f"  Sequence S_j: {[wp.wid for wp in seq] if seq else 'No waypoints assigned'}")
        print(f"  m_j (repetitions): {uav.m_j}")
        print(f"  Tour T_j (indices): {[wp.wid if isinstance(wp, Waypoint) else environment.depot for wp in tour]}")
        print(f"  Tour flight time T_j (s): {tour_time:.2f}")
        print(f"  Remaining time (s): {remaining_time:.2f}")

    if unassigned_targets:
        print(f"\nUnassigned targets: {sorted([wp.wid for wp in unassigned_targets])}")
    else:
        print("\nAll targets were assigned to at least one UAV (m_j >= 1).")


# ============================================================
# Excel export helpers (one row per UAV, one sheet per simulation)
# ============================================================

def prepare_output_dirs(base_dir: str) -> Tuple[str, str]:
    """
    Prepare the output directories for saving revenue and sequence Excel files. The structure will be:
    - base_dir/
        - revenue/
            - YYYY-MM-DD/
                - simulation_1/
                - simulation_2/
                - ...
        - sequences/
            - YYYY-MM-DD/
                - simulation_1/
                - simulation_2/
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    revenue_base = os.path.join(base_dir, "revenue", date_str)
    sequences_base = os.path.join(base_dir, "sequences", date_str)

    os.makedirs(revenue_base, exist_ok=True)
    os.makedirs(sequences_base, exist_ok=True)

    existing = [
        d for d in os.listdir(revenue_base)
        if os.path.isdir(os.path.join(revenue_base, d)) and d.startswith("simulation_")
    ]
    sim_idx = len(existing) + 1

    revenue_dir = os.path.join(revenue_base, f"simulation_{sim_idx}")
    sequences_dir = os.path.join(sequences_base, f"simulation_{sim_idx}")

    os.makedirs(revenue_dir, exist_ok=True)
    os.makedirs(sequences_dir, exist_ok=True)

    return revenue_dir, sequences_dir


def export_runs_to_excel(
    m: int,
    uav_speed: float,
    max_flight_time: float,
    grid_size: int,
    rev_sheets: List[pd.DataFrame],
    seq_sheets: List[pd.DataFrame],
    revenue_dir: str,
    sequences_dir: str,
) -> Tuple[str, str]:
    """
    Export the revenue and sequence data for multiple simulation runs to separate Excel files.
    Each file will contain one sheet per simulation run, with the sheet name indicating the run number (e.g., "SimRun1", "SimRun2", etc.).
    The revenue file will have columns for each UAV's revenue, while the sequence file will have columns for each UAV's assigned sequence and m_j value.
    """
    rev_path = os.path.join(
        revenue_dir,
        f"UAVs{m}_GRID{grid_size}_Greedy.xlsx"
    )
    seq_path = os.path.join(
        sequences_dir,
        f"UAVs{m}_GRID{grid_size}_{max_flight_time}_{uav_speed}_Greedy_sequences.xlsx"
    )

    with pd.ExcelWriter(rev_path) as writer:
        for idx, df in enumerate(rev_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    with pd.ExcelWriter(seq_path) as writer:
        for idx, df in enumerate(seq_sheets, start=1):
            df.to_excel(writer, sheet_name=f"SimRun{idx}", index=False)

    return rev_path, seq_path


def run_simulation(params: dict, waypoint_files: List[Path]) -> None:
    """
    Run the simulation using external waypoint files. This function is called when USE_EXTERNAL_WAYPOINTS is True.
    It will process each waypoint file, extract the number of UAVs and grid size, load the waypoints from each sheet, run the greedy allocator, and save the results to Excel files.
    """
    uav_speed = params["uav_speed"]
    max_flight_time = params["max_flight_time"]

    for waypoint_file in waypoint_files:
        m = extract_num_uavs(str(waypoint_file))
        grid_size = extract_grid_size(str(waypoint_file))
        workbook = pd.ExcelFile(waypoint_file)

        rev_sheets: List[pd.DataFrame] = []
        seq_sheets: List[pd.DataFrame] = []

        print(f"\n=== Processing waypoint file: {waypoint_file} (m = {m}), grid_size = {grid_size} ===")

        for sheet_name in workbook.sheet_names:
            generated_targets = load_waypoints_sheet(str(waypoint_file), sheet_name)
            environment = GridEnvironment(
                external_targets=generated_targets,
                depot_location=DEPOT_LOCATION,
                base_revenue=params["base_revenue"],
                base_seed=params["base_seed"],
                use_fixed_targets=params.get("use_fixed_targets", True),
                fixed_target_indices=params.get("fixed_target_indices", [2, 3, 5, 6, 7]),
                grid_width=params.get("grid_width", grid_size),
                grid_height=params.get("grid_height", grid_size),
                grid_spacing=params.get("grid_spacing", 1.0),
                min_wp_revenue=params.get("min_wp_revenue", 60.0),
                max_wp_revenue=params.get("max_wp_revenue", 600.0),
                num_targets=params.get("num_targets", 5),
            )

            allocator = GreedyAllocator(
                environment=environment,
                num_uavs=m,
                uav_speed=uav_speed,
                max_flight_time=max_flight_time,
            )
            uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

            # Greedy has no negotiation rounds, so store a single row with round 0
            rev_row = {"negotiation_round": 0}
            seq_row = {"negotiation_round": 0}

            for uav in uavs:
                z_j = allocator.compute_revenue_rate(uav)
                seq_ids = [wp.wid for wp in uav.sequence] if uav.sequence else []
                seq_str = "-".join(map(str, seq_ids))

                rev_row[f"UAV{uav.uid}"] = z_j
                seq_row[f"UAV{uav.uid}"] = seq_str
                seq_row[f"m_{uav.uid}"] = uav.m_j

            rev_sheets.append(pd.DataFrame([rev_row]))
            seq_sheets.append(pd.DataFrame([seq_row]))

        revenue_dir, sequences_dir = prepare_output_dirs(OUTPUT_BASE_DIR)
        revenue_file, sequences_file = export_runs_to_excel(
            m=m,
            uav_speed=uav_speed,
            max_flight_time=max_flight_time,
            grid_size=grid_size,
            rev_sheets=rev_sheets,
            seq_sheets=seq_sheets,
            revenue_dir=revenue_dir,
            sequences_dir=sequences_dir,
        )

        print(f"Saved revenue Excel:   {revenue_file}")
        print(f"Saved sequences Excel: {sequences_file}")


def run_example(params: dict) -> None:
    """
    Run the example simulation using a default grid environment. This function is called when USE_EXTERNAL_WAYPOINTS is False.
    It will build a default grid environment, assign random revenues to the target waypoints, run the greedy allocator, and print the results to the console.
    """
    environment = GridEnvironment(
        external_targets=None,
        depot_location=DEPOT_LOCATION,
        base_revenue=params["base_revenue"],
        base_seed=params["base_seed"],
        use_fixed_targets=params["use_fixed_targets"],
        fixed_target_indices=params["fixed_target_indices"],
        grid_width=params["grid_width"],
        grid_height=params["grid_height"],
        grid_spacing=params["grid_spacing"],
        min_wp_revenue=params["min_wp_revenue"],
        max_wp_revenue=params["max_wp_revenue"],
        num_targets=params.get("num_targets", 5),
    )
    environment.assign_random_revenues()

    print("\n=== Default Grid Environment ===")
    environment.print_summary()

    allocator = GreedyAllocator(
        environment=environment,
        num_uavs=params["num_uavs"],
        uav_speed=params["uav_speed"],
        max_flight_time=params["max_flight_time"],
    )
    uavs, unassigned_targets, total_revenue, total_revenue_rate = allocator.solve()

    print_solution(
        environment=environment,
        allocator=allocator,
        uavs=uavs,
        unassigned_targets=unassigned_targets,
        total_revenue=total_revenue,
        total_revenue_rate=total_revenue_rate,
        header="Greedy Result with Default Grid",
    )


def load_configuration(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def setup_parameters(config: dict, use_external_waypoints: bool) -> dict:
    uav = config.get("uav", {})
    simulation = config.get("simulation", {})
    grid = config.get("grid", {})
    revenue = config.get("revenue", {})

    params = {
        "uav_speed": uav.get("speed", 16.0),
        "max_flight_time": uav.get("max_flight_time", 1920.0),
        "base_revenue": revenue.get("fixed_value", 30.0),
        "base_seed": simulation.get("seed", 32),
        "num_uavs": uav.get("num_uavs", 2),
    }

    if not use_external_waypoints:
        params.update(
            {
                "use_fixed_targets": True,
                "fixed_target_indices": [2, 3, 5, 6, 7],
                "grid_width": grid.get("width", 3),
                "grid_height": grid.get("height", 3),
                "grid_spacing": grid.get("spacing", 1.0),
                "min_wp_revenue": revenue.get("min", 60.0),
                "max_wp_revenue": revenue.get("max", 600.0),
                # default if you ever switch use_fixed_targets to False
                "num_targets": 5,
            }
        )

    return params

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config = load_configuration(SETTINGS_PATH)
    params = setup_parameters(config, use_external_waypoints=USE_EXTERNAL_WAYPOINTS)
    simple_cfg = load_simple_config(config)

    random.seed(params["base_seed"])

    waypoint_files = sorted(WAYPOINTS_PATH.rglob("*_waypoints.xlsx"))

    if USE_EXTERNAL_WAYPOINTS:
        if not waypoint_files:
            raise FileNotFoundError(f"No waypoint Excel files found under: {WAYPOINTS_PATH}")
        run_simulation(params=params, waypoint_files=waypoint_files)

        # NEW: generate GIFs with blockspots
        generate_blockspot_gifs(
            results_base_dir=OUTPUT_BASE_DIR,
            waypoints_root=WAYPOINTS_PATH,
            cfg=simple_cfg,
        )
    else:
        run_example(params=params)
