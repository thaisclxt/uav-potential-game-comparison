import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from pathlib import Path
from typing import Optional
from matplotlib.animation import PillowWriter

from .utils import extract_grid_size, extract_num_uavs


def generate_uav_contribution_boxplots(outputs_dir: Path) -> None:
    """
    For each scenario folder like:
      outputs/UAVs{m}_GRID{grid}/

    Read the Greedy revenue workbook:
      revenue/UAVs{m}_GRID{grid}_Greedy.xlsx

    For each UAV j, compute its contribution share in each simulation run:
      share_j = (UAVj revenue at final round) / (sum_k UAVk revenue at final round)

    Then build ONE boxplot per scenario showing the distribution of share_j
    across all simulation runs, with x-axis labels:
      UAV0, UAV1, ..., UAV{m-1}

    Save the figure as:
      visualizations/uav_contribution_greedy.png
    """
    outputs_dir = Path(outputs_dir)

    if not outputs_dir.exists():
        print(f"[BOX] Output directory does not exist: {outputs_dir}")
        return

    scenario_dirs = sorted(
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.startswith("UAVs")
    )

    if not scenario_dirs:
        print(f"[BOX] No scenario folders found in {outputs_dir}")
        return

    for scenario_dir in scenario_dirs:
        revenue_dir = scenario_dir / "revenue"
        visualizations_dir = scenario_dir / "visualizations"

        if not revenue_dir.exists():
            print(f"[BOX] Missing revenue dir: {revenue_dir}")
            continue

        # One Greedy revenue file per scenario
        revenue_files = sorted(revenue_dir.glob("UAVs*_GRID*_Greedy.xlsx"))
        if not revenue_files:
            print(f"[BOX] No Greedy revenue files found in {revenue_dir}")
            continue

        # If there are multiple, we'll just use the first (your structure likely has one)
        revenue_file = revenue_files[0]

        try:
            sheets = pd.read_excel(revenue_file, sheet_name=None, index_col=0)
        except Exception as exc:
            print(f"[BOX] Could not read {revenue_file}: {exc}")
            continue

        # Collect per-UAV contribution shares over all simulation runs
        per_uav_shares: list[list[float]] = []
        uav_labels: list[str] = []

        # We will infer the UAV columns once from the first valid sheet
        first_uav_cols: Optional[list[str]] = None

        for run_name, df in sheets.items():
            uav_cols = [c for c in df.columns if str(c).upper().startswith("UAV")]
            if not uav_cols:
                print(f"[BOX] Skipping sheet {run_name}: no UAV columns found")
                continue

            if first_uav_cols is None:
                first_uav_cols = uav_cols
                uav_labels = [str(c) for c in uav_cols]
                per_uav_shares = [[] for _ in uav_cols]
            else:
                # Ensure consistent UAV columns across sheets
                if uav_cols != first_uav_cols:
                    print(
                        f"[BOX] Skipping sheet {run_name}: UAV columns {uav_cols} "
                        f"do not match first sheet {first_uav_cols}"
                    )
                    continue

            # Use final row (last index) as "final" revenue rate
            final_row = df[uav_cols].iloc[-1]
            total = float(final_row.sum())
            if total <= 0:
                print(
                    f"[BOX] Skipping sheet {run_name}: non-positive total revenue "
                    f"(total={total:.4f})"
                )
                continue

            # Compute shares for this run and append
            for idx, col in enumerate(uav_cols):
                share = float(final_row[col]) / total
                per_uav_shares[idx].append(share)

        if not per_uav_shares or all(len(v) == 0 for v in per_uav_shares):
            print(f"[BOX] No valid UAV contribution data found in {revenue_file.name}")
            continue

        visualizations_dir.mkdir(parents=True, exist_ok=True)
        out_path = visualizations_dir / "uav_contribution_greedy.png"

        # Create boxplot: one box per UAV, showing distribution over runs
        fig, ax = plt.subplots(figsize=(5, 6))

        bp = ax.boxplot(
            per_uav_shares,
            tick_labels=uav_labels,
            patch_artist=True,
        )

        # Style boxes
        for box in bp["boxes"]:
            box.set_facecolor("C0")
            box.set_edgecolor("black")

        for median in bp["medians"]:
            median.set(color="orange", linewidth=2)

        ax.set_ylabel("Share of total revenue rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[BOX] Saved UAV contribution boxplot to {out_path}")


def _algo_label_from_seq_file(seq_file: Path) -> Optional[str]:
    """
    Infer algorithm label from the sequence file name.

    Examples:
      UAVs3_GRID10_IRADA_sequences.xlsx       -> IRADA
      UAVs3_GRID10_ModeX_Y_sequences.xlsx     -> ModeX_Y
    """
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


def get_revenue_file_for_sequence(seq_file: Path, rev_dir: Path) -> Path | None:
    """
    Given a sequence file, find the matching revenue Excel file in rev_dir.

    Matching is based on:
      - UAVs{m}_GRID{grid_size} prefix
      - optional algorithm label if present in filename
    """
    stem = seq_file.stem.replace("_sequences", "")
    parts = stem.split("_")

    if len(parts) < 2:
        return None

    prefix = f"{parts[0]}_{parts[1]}"

    matches = sorted(rev_dir.glob(f"{prefix}_*.xlsx"))
    if not matches:
        matches = sorted(rev_dir.glob(f"{prefix}.xlsx"))

    if not matches:
        return None

    seq_algo = (_algo_label_from_seq_file(seq_file) or "").lower()

    if seq_algo:
        for candidate in matches:
            if seq_algo in candidate.stem.lower():
                return candidate

    return matches[0]


def _generate_gif_for_run(
    seq_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    wp_df: pd.DataFrame,
    out_path: Path,
    run_name: str,
    algo_label: str,
    n_uavs: int,
    grid_dim: int,
    max_flight_time: float,
) -> None:
    """
    Generate one GIF for one simulation run.

    A simulation run corresponds to one sheet, such as:
    - SimRun1
    - SimRun2
    - SimRun3
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zero_color = "#CCCCCC"
    nonzero_color = "#111111"

    n_seq = len(seq_df)
    n_rev = len(rev_df)

    if n_seq == 0:
        print(f"[GIF] Skipping empty sequence data for run {run_name}")
        return

    if n_rev < n_seq:
        last_row = rev_df.iloc[-1]
        extra = pd.DataFrame([last_row] * (n_seq - n_rev), columns=rev_df.columns)
        rev_df = pd.concat([rev_df, extra], ignore_index=True)
        print(
            f"[GIF] Extended revenue {out_path.stem}:{run_name} "
            f"from {n_rev} to {n_seq} rows."
        )
    elif n_rev > n_seq:
        rev_df = rev_df.iloc[:n_seq].copy()
        print(
            f"[GIF] Truncated revenue {out_path.stem}:{run_name} "
            f"from {n_rev} to {n_seq} rows."
        )

    seq_df = seq_df.reset_index(drop=True)
    rev_df = rev_df.reset_index(drop=True)

    coords = {
        int(r.Waypoint): (float(r.X), float(r.Y), float(r.Revenue))
        for _, r in wp_df.iterrows()
    }

    if not coords:
        print(f"[GIF] Skipping {run_name}: no waypoint coordinates found.")
        return

    xs_sorted = [coords[i][0] for i in sorted(coords)]
    d = abs(xs_sorted[1] - xs_sorted[0]) if len(xs_sorted) >= 2 else 1.0

    xs_all = [c[0] for c in coords.values()]
    ys_all = [c[1] for c in coords.values()]

    fig = plt.figure(figsize=(6, 5))

    header = (
        f"{algo_label} UAVs = {n_uavs} Grid = {grid_dim} × {grid_dim}\n"
        f"Simulation Run = {run_name}"
    )
    fig.text(0.5, 0.98, header, ha="center", va="top", fontsize=10)

    ax = fig.add_axes([0.05, 0.1, 0.7, 0.8])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    for _, (x, y, rev_val) in coords.items():
        rect = plt.Rectangle(
            (x - 0.5 * d, y - 0.5 * d),
            d,
            d,
            facecolor=zero_color if rev_val <= 0 else nonzero_color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    ax.set_xlim(min(xs_all) - d, max(xs_all) + d)
    ax.set_ylim(min(ys_all) - d, max(ys_all) + d)

    uav_cols = [c for c in seq_df.columns if str(c).upper().startswith("UAV")]
    uav_colors = {j: colors[j % len(colors)] for j in range(len(uav_cols))}

    paths = {
        j: ax.plot([], [], color=uav_colors[j], linewidth=1.0)[0]
        for j in range(len(uav_cols))
    }

    def frame_to_paths(frame: int) -> None:
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
        t = frame * max_flight_time / max(1, len(seq_df) - 1)
        time_text.set_text(f"Round {frame}\nTime ≈ {t:.1f}s")

        tot = 0.0
        for c in rev_df.columns:
            if str(c).upper().startswith("UAV"):
                val = rev_df.iloc[frame][c]
                if pd.notna(val):
                    tot += float(val)

        rev_text.set_text(f"Total revenue rate:\n{tot:.1f}")
        return list(paths.values()) + [time_text, rev_text]

    frames_idx = range(0, len(seq_df), 2)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames_idx,
        interval=300,
        blit=False,
        repeat=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GIF] Saving {out_path}")
    writer = PillowWriter(fps=3)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)


def _generate_gifs_for_sequence_file(
    sequence_file: Path,
    revenue_file: Path,
    waypoint_file: Path,
    visualizations_dir: Path,
    max_flight_time: float,
) -> None:
    """
    Generate GIFs for each simulation run in one sequence Excel file.

    Example:
    - sequence_file has sheets SimRun1, SimRun2, SimRun3
    - revenue_file has matching sheets SimRun1, SimRun2, SimRun3
    - waypoint_file has matching sheets SimRun1, SimRun2, SimRun3

    This function loops over each simulation run sheet and creates one GIF per run.
    """
    base_stem = sequence_file.stem.replace("_sequences", "")
    algo_label = _algo_label_from_seq_file(sequence_file) or "Greedy"

    try:
        n_uavs = extract_num_uavs(sequence_file.name)
        grid_dim = extract_grid_size(sequence_file.name)
    except Exception as exc:
        print(f"[GIF] Skipping {sequence_file.name}: {exc}")
        return

    seq_sheets = pd.read_excel(sequence_file, sheet_name=None)
    rev_sheets = pd.read_excel(revenue_file, sheet_name=None)
    waypoint_book = pd.ExcelFile(waypoint_file)

    cfg_gifs_dir = visualizations_dir / algo_label
    cfg_gifs_dir.mkdir(parents=True, exist_ok=True)

    for run_name, seq_df in seq_sheets.items():
        rev_df = rev_sheets.get(run_name)
        if rev_df is None:
            print(f"[GIF] Missing revenue sheet for run {run_name} in {revenue_file.name}")
            continue

        if run_name not in waypoint_book.sheet_names:
            print(f"[GIF] Missing waypoint sheet {run_name} in {waypoint_file.name}")
            continue

        wp_df = pd.read_excel(waypoint_file, sheet_name=run_name)
        gif_name = f"{base_stem}_{run_name}.gif"
        out_path = cfg_gifs_dir / gif_name

        _generate_gif_for_run(
            seq_df=seq_df,
            rev_df=rev_df,
            wp_df=wp_df,
            out_path=out_path,
            run_name=run_name,           # e.g. "SimRun1"
            algo_label=algo_label,
            n_uavs=n_uavs,
            grid_dim=grid_dim,
            max_flight_time=max_flight_time,
        )


def generate_blockspot_gifs(
    outputs_dir: Path,
    waypoints_dir: Path,
    uav_cfg,
) -> None:
    """
    Walk through outputs/UAVsM_GRIDN and generate GIFs.
    """
    outputs_dir = Path(outputs_dir)
    waypoints_dir = Path(waypoints_dir)

    if not outputs_dir.exists():
        print(f"[VIS] Output directory does not exist: {outputs_dir}")
        return

    scenario_dirs = sorted(
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.startswith("UAVs")
    )

    if not scenario_dirs:
        print(f"[VIS] No scenario folders found in {outputs_dir}")
        return

    for scenario_dir in scenario_dirs:
        sequences_dir = scenario_dir / "sequences"
        revenue_dir = scenario_dir / "revenue"
        visualizations_dir = scenario_dir / "visualizations"

        if not sequences_dir.exists():
            print(f"[VIS] Missing sequences dir: {sequences_dir}")
            continue

        if not revenue_dir.exists():
            print(f"[VIS] Missing revenue dir: {revenue_dir}")
            continue

        visualizations_dir.mkdir(parents=True, exist_ok=True)

        sequence_files = sorted(sequences_dir.glob("*_sequences.xlsx"))
        if not sequence_files:
            print(f"[VIS] No sequence files found in {sequences_dir}")
            continue

        for seq_file in sequence_files:
            try:
                m = extract_num_uavs(seq_file.name)
                grid_size = extract_grid_size(seq_file.name)
            except Exception as exc:
                print(f"[VIS] Skipping {seq_file.name}: {exc}")
                continue

            revenue_file = get_revenue_file_for_sequence(seq_file, revenue_dir)
            if revenue_file is None or not revenue_file.exists():
                print(f"[VIS] No matching revenue file for {seq_file.name}")
                continue

            wp_pattern = f"UAVs{m}_GRID{grid_size}_waypoints.xlsx"
            wp_matches = sorted(waypoints_dir.rglob(wp_pattern))
            if not wp_matches:
                print(
                    f"[VIS] Skipping {seq_file.name}: "
                    f"no waypoint file found matching {wp_pattern}"
                )
                continue

            waypoint_file = wp_matches[0]

            _generate_gifs_for_sequence_file(
                sequence_file=seq_file,
                revenue_file=revenue_file,
                waypoint_file=waypoint_file,
                visualizations_dir=visualizations_dir,
                max_flight_time=uav_cfg.max_flight_time,
            )