import random, math

GRID_SIZE = 13
WAYPOINT_SPACING = 50.0

TOTAL_TARGETS = 21
M_MAX = 6                  # Maximum number of UAVs considered in Algorithm 1 (M)

UAV_SPEED = 10.0           # Velocity V (distance units per second)
MAX_FLIGHT_TIME = 1800.0   # Maximum flight time Tf_max(k) (seconds) for all UAVs

O_OPERATORS = 3            # Number of UAV operators O (for waiting time model)
TP_PREP = 300.0            # Preparation time Tp (seconds) for one UAV

MIN_WP_REVENUE = 60.0      # Minimum revenue for a target waypoint
MAX_WP_REVENUE = 600.0     # Maximum revenue for a target waypoint


class Waypoint:
    """
    Node in the 2D grid.

    Attributes:
        id (int): unique waypoint index.
        x, y (float): coordinates in 2D plane.
        revenue (float): revenue gained if this waypoint is visited
                         (used only for target waypoints, depot has 0).
    """
    def __init__(self, wid, x, y):
        self.id = wid
        self.x = x
        self.y = y
        self.revenue = 0.0

    def update_revenue(self):
        """
        Assign a random revenue value to this waypoint within the global range.
        """
        self.revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)


class UAV:
    """
    UAV model.

    Attributes:
        id (int): UAV index (1..m).
        route (list[Waypoint]): ordered list of waypoints visited by this UAV.
                                The first waypoint is always the depot P0.
    """
    def __init__(self, uid, start_wp):
        self.id = uid
        self.route = [start_wp]


def build_grid():
    """
    Build a GRID_SIZE x GRID_SIZE grid of waypoints spaced by WAYPOINT_SPACING.

    Returns:
        list[Waypoint]: all waypoints, indexed by their id (0..N-1).
                        Waypoint 0 is used as the depot P0.
    """
    waypoints = []
    wid = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = i * WAYPOINT_SPACING
            y = j * WAYPOINT_SPACING
            waypoints.append(Waypoint(wid, x, y))
            wid += 1
    return waypoints


def distance(wp1, wp2):
    """
    Euclidean distance between two waypoints.
    """
    dx = wp1.x - wp2.x
    dy = wp1.y - wp2.y
    return math.hypot(dx, dy)


def travel_time(wp1, wp2):
    """
    Flight time between two waypoints
    based on constant speed UAV_SPEED.
    """
    return distance(wp1, wp2) / UAV_SPEED


def flight_time(route):
    """
    Compute T_f(k): flight time of a UAV along its current route.

    It is the sum of travel_time over all consecutive waypoint pairs
    in the route. This corresponds to the flight time definition in
    the paper (Definition 2) but omits the turning-angle part.

    Args:
        route (list[Waypoint]): current path of a UAV.

    Returns:
        float: total flight time along the route.
    """
    t = 0.0
    for i in range(len(route) - 1):
        t += travel_time(route[i], route[i+1])
    return t


def waiting_time(k):
    """
    Compute T_w(k): waiting time before takeoff for the k-th UAV.

    Assumptions (from Definition 1):
    - There are O operators.
    - Each UAV requires Tp seconds of preparation.
    - Up to O UAVs can be prepared in parallel in each "batch".

    Example: O=2, Tp=5 min
      U1,U2 wait 5 min, U3,U4 wait 10 min, U5,U6 wait 15 min, ...

    Args:
        k (int): UAV index (1-based).
        Tp (float): preparation time for one UAV.
        O (int): number of operators.

    Returns:
        float: waiting time for the k-th UAV.
    """
    # How many preparation batches have been completed before UAV k?
    batch_index = (k - 1) // O_OPERATORS
    return (batch_index + 1) * TP_PREP


def cumulative_time(uav):
    """
    Compute T_c(k) = T_w(k) + T_f(k): cumulative time of UAV k.

    This is used in the greedy allocation to pick the UAV with
    the smallest current load (shortest cumulative time).

    Args:
        uav (UAV): UAV whose cumulative time is computed.
        Tp (float): preparation time per UAV.
        O (int): number of operators.

    Returns:
        float: cumulative time for this UAV.
    """
    k = uav.id
    Twk = waiting_time(k)
    Tf_k = flight_time(uav.route)
    return Twk + Tf_k


def revenue_function(uavs):
    """
    Compute total revenue of a given assignment (set of UAV routes).

    Revenue of a route is defined as the sum of revenues of all target
    waypoints visited by that UAV. The depot P0 (waypoint 0) is excluded.

    Args:
        uavs (list[UAV]): UAVs with completed routes.

    Returns:
        float: total revenue over all UAVs.
    """
    total_rev = 0.0
    for uav in uavs:
        # Skip the first waypoint (depot) in each route
        for wp in uav.route[1:]:
            total_rev += wp.revenue
    return total_rev


def find_Mmin(waypoints, target_indices):
    """
    Phase 1 of Algorithm 1: find the minimal number of UAVs Mmin
    that can cover all targets under the flight time constraint.

    Logic (adapted from Algorithm 1, steps 2–15):
      - Start with k = 1.
      - For UAV Uk, repeatedly assign it the nearest feasible target Pc
        (feasible means adding Pc keeps Tf(k) <= Tfmax(k)).
      - When Uk cannot take any more targets, move to UAV U(k+1).
      - Stop when all targets are assigned or k > M.
      - The resulting k is taken as Mmin.

    Args:
        waypoints (list[Waypoint]): full grid of waypoints (including depot).
        target_indices (list[int]): indices of target waypoints to be covered.
        M (int): maximum number of UAVs available.
        Tfmax_value (float): maximum flight time allowed for each UAV.

    Returns:
        tuple:
            Mmin (int): minimal number of UAVs needed to cover all targets.
            uavs_Mmin (list[UAV]): UAV objects with their routes from Phase 1.
    """
    depot = waypoints[0]
    # Create M UAVs at the depot; we will use only UAVs 1..k
    uavs = [UAV(uid=i+1, start_wp=depot) for i in range(M_MAX)]

    remaining = set(target_indices)
    k = 1  # current UAV index (1-based)

    while remaining and k <= M_MAX:
        uk = uavs[k-1]              # UAV Uk
        last_wp = uk.route[-1]

        # Find all targets that Uk can still reach without exceeding Tfmax
        feasible = []
        Tf_k = flight_time(uk.route)
        for idx in remaining:
            wp = waypoints[idx]
            extra_out = travel_time(last_wp, wp)
            extra_back = travel_time(wp, depot)
            if Tf_k + extra_out + extra_back <= MAX_FLIGHT_TIME:
                feasible.append(idx)

        if feasible:
            # Among feasible targets, choose the nearest one to Uk
            closest_idx = min(
                feasible,
                key=lambda i: distance(last_wp, waypoints[i])
            )
            closest_wp = waypoints[closest_idx]
            # Assign this target Pc to Uk and update its route
            uk.route.append(closest_wp)
            remaining.remove(closest_idx)
        else:
            # Uk cannot take more targets under Tfmax, move to UAV U(k+1)
            k += 1

    Mmin = k
    uavs_Mmin = uavs[:Mmin]
    return Mmin, uavs_Mmin


def greedy_task_assignment_phase2(uavs, waypoints, target_indices):
    """
    Phase 2 core of Algorithm 1 for a fixed number of UAVs m.

    For a given set of m UAVs:
      - Repeatedly select the UAV Us with the shortest cumulative time T_c(k).
      - For Us, among all remaining targets, find those that keep Tf(k)
        <= Tfmax(k) if assigned, then choose the nearest such target Pc.
      - Assign Pc to Us and update its route.
      - Stop when no remaining targets are left or no feasible assignments exist.

    This implements steps 17–23 (task assignment) for a fixed m.

    Args:
        uavs (list[UAV]): list of m UAVs starting at the depot.
        waypoints (list[Waypoint]): full grid of waypoints.
        target_indices (list[int]): indices of target waypoints to be covered.
        Tfmax_value (float): maximum flight time allowed for each UAV.

    Returns:
        list[UAV]: the same UAV list, with updated routes after greedy assignment.
    """
    remaining = set(target_indices)

    while remaining:
        # Select Us: the UAV with minimum cumulative time T_c(k)
        us = min(
            uavs,
            key=lambda u: cumulative_time(u)
        )

        last_wp = us.route[-1]

        # Compute which remaining targets are feasible for Us under Tfmax
        feasible = []
        Tf_s = flight_time(us.route)
        for idx in remaining:
            wp = waypoints[idx]
            extra_out = travel_time(last_wp, wp)
            extra_back = travel_time(wp, depot)
            if Tf_s + extra_out + extra_back <= MAX_FLIGHT_TIME:
                feasible.append(idx)

        if not feasible:
            # This UAV cannot take any of the remaining targets;
            # in this implementation we stop the entire allocation.
            # (You could extend this to try other UAVs instead.)
            break

        # Among feasible targets, choose the nearest one to Us
        closest_idx = min(
            feasible,
            key=lambda i: distance(last_wp, waypoints[i])
        )
        closest_wp = waypoints[closest_idx]

        # Assign Pc to Us and update its route
        us.route.append(closest_wp)
        remaining.remove(closest_idx)

    # After this loop, uavs encode an allocation scheme A_m for this m.
    return uavs

def greedy_allocation_algorithm(waypoints, target_indices):
    """
    Full greedy allocation algorithm based on Algorithm 1,
    with a revenue-based objective instead of the paper's cost.

    Phase 1:
        - Find the minimal number of UAVs Mmin required to cover all targets
          given the flight time constraint Tfmax. (Feasibility step.)

    Phase 2:
        - For each m in [Mmin, M], run the greedy task assignment (Phase 2 core).
        - Compute total revenue for that m.
        - Keep the m and assignment that maximize total revenue.

    Args:
        waypoints (list[Waypoint]): full grid of waypoints.
        target_indices (list[int]): indices of target waypoints to be covered.
        M (int): maximum number of UAVs considered.
        Tfmax_value (float): maximum flight time allowed for each UAV.

    Returns:
        tuple:
            optimal_m (int): number of UAVs m that yields maximum revenue.
            optimal_assignment (list[list[int]]):
                list of routes, each route is a list of waypoint IDs for one UAV.
            optimal_revenue (float): maximum revenue achieved by the selected m.
    """
    # Phase 1: determine Mmin by feasibility (ignoring revenue)
    Mmin, _ = find_Mmin(waypoints, target_indices)

    best_m = Mmin
    best_rev = -float('inf')
    best_assignment = None

    depot = waypoints[0]

    # Phase 2: for each m in [Mmin, M], perform greedy assignment and evaluate revenue
    for m in range(Mmin, M_MAX + 1):
        # Initialize m UAVs at the depot
        uavs_m = [UAV(uid=i+1, start_wp=depot) for i in range(m)]

        # Greedy task assignment for this m
        uavs_m = greedy_task_assignment_phase2(uavs_m, waypoints, target_indices)

        # Compute total revenue for the resulting assignment
        R_m = revenue_function(uavs_m)

        # Keep the best (highest revenue) solution found so far
        if R_m >= best_rev:
            best_rev = R_m
            best_m = m
            # Store routes as lists of waypoint IDs for easier inspection/printing
            best_assignment = [
                [wp.id for wp in uav.route]
                for uav in uavs_m
            ]

    return best_m, best_assignment, best_rev


if __name__ == "__main__":
    # Build environment and get depot waypoint P0
    waypoints = build_grid()
    depot = waypoints[0]

    # Fixed values just for testing
    target_indices = [8, 168, 20, 119, 12, 135, 111, 69, 163, 100, 72, 46, 159, 104, 115, 145, 125, 27, 71, 44, 139]
    target_revenues = [
        534.27,
        271.88,
        593.41,
        146.52,
        415.09,
        367.43,
        492.75,
        88.64,
        577.32,
        329.18,
        212.97,
        451.60,
        189.34,
        560.21,
        305.47,
        97.85,
        248.63,
        384.19,
        140.72,
        520.56,
        276.04
    ]

    for idx, rev in zip(target_indices, target_revenues):
        wp = waypoints[idx]
        wp.revenue = rev
        print(f"Target {wp.id}: ({wp.x}, {wp.y}), revenue = {wp.revenue:.2f}")

    # Randomly choose TOTAL_TARGETS target waypoints (excluding depot 0)
    # target_indices = random.sample(range(1, len(waypoints)), TOTAL_TARGETS)

    # print("Target waypoints and revenues:")
    # for i in target_indices:
    #     wp = waypoints[i]
    #     wp.update_revenue()  # assign random revenue to this target
    #     print(f"Target {wp.id}: ({wp.x}, {wp.y}), revenue = {wp.revenue:.2f}")

    # Run greedy allocation algorithm with revenue objective
    optimal_m, optimal_assignment, optimal_revenue = greedy_allocation_algorithm(
        waypoints=waypoints,
        target_indices=target_indices,
    )

    print(f"\nOptimal number of UAVs (m): {optimal_m}")
    print(f"Optimal revenue: {optimal_revenue:.2f}")
    print("Optimal assignment (routes by waypoint IDs):")
    if optimal_assignment is not None:
        for i, route in enumerate(optimal_assignment):
            print(f"UAV {i+1} route: {route}")