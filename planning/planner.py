from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = Queue()
    frontier.push((start, []))
    visited = {start}

    while not frontier.isEmpty():
        state, actions = frontier.pop()
        for next_state, action, _ in problem.getSuccessors(state):
            if next_state in visited:
                continue
            new_actions = actions + [action]
            if problem.isGoalState(next_state):
                return new_actions
            visited.add(next_state)
            frontier.push((next_state, new_actions))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    if not (action.add_list & goal_set):
        return None
    if action.del_list & goal_set:
        return None
    return (goal_set - action.add_list) | action.precond_pos


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    initial = problem.initial_state
    goal = problem.goal

    if goal.issubset(initial):
        return []

    static_predicates = {"MedicalPost", "Adjacent", "Pickable"}

    def is_dead_end(goal_set: State) -> bool:
        robot_at_cells = []
        holding_objs = []
        for f in goal_set:
            if f[0] in static_predicates:
                return True
            if f[0] == "At" and len(f) == 3 and f[1] == "robot":
                robot_at_cells.append(f[2])
            if f[0] == "Holding":
                holding_objs.append(f[2])
        if len(robot_at_cells) > 1:
            return True
        if len(holding_objs) > 1:
            return True
        if holding_objs and ("HandsFree", "robot") in goal_set:
            return True
        if robot_at_cells and ("Free", robot_at_cells[0]) in goal_set:
            return True
        return False

    from collections import defaultdict
    all_actions = get_all_groundings(problem.domain, problem.objects)

    add_index: dict = defaultdict(list)
    for _a in all_actions:
        for _f in _a.add_list:
            add_index[_f].append(_a)

    def visited_key(goal_set: State) -> State:
        """
        Deduplication key: only the UNSATISFIED fluents (not in initial state),
        minus static predicates already handled by dead-end pruning.
        Two goal sets that share the same unsatisfied core represent equivalent
        search states for BFS-optimal planning: they differ only in background
        fluents that are always true in the initial state and need not be achieved.
        The full goal is still used for DEL ∩ g checks during regression.
        """
        return frozenset(f for f in goal_set if f not in initial)

    start_key = visited_key(goal)
    if not start_key:
        return []

    frontier = Queue()
    frontier.push((goal, []))
    visited: set[State] = {start_key}

    while not frontier.isEmpty():
        current_goal, actions = frontier.pop()
        problem._expanded += 1

        unsatisfied = current_goal - initial
        seen_actions: set = set()
        for fluent in unsatisfied:
            for action in add_index.get(fluent, []):
                if id(action) in seen_actions:
                    continue
                seen_actions.add(id(action))

                regressed = regress(current_goal, action)
                if regressed is None:
                    continue

                simplified = frozenset(
                    f for f in regressed
                    if f not in initial or f[0] not in static_predicates
                )

                if is_dead_end(simplified):
                    continue

                new_actions = [action] + actions

                if simplified.issubset(initial):
                    return new_actions

                key = visited_key(simplified)
                if key not in visited:
                    visited.add(key)
                    frontier.push((simplified, new_actions))

    return []


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    goal = problem.goal
    domain = problem.domain
    objects = problem.objects

    frontier = PriorityQueue()
    h0 = heuristic(start, goal, domain, objects)
    frontier.push((start, []), h0)
    best_g: dict[State, float] = {start: 0}

    while not frontier.isEmpty():
        state, actions = frontier.pop()
        g = len(actions)

        if g > best_g.get(state, float("inf")):
            continue

        if problem.isGoalState(state):
            return actions

        for next_state, action, cost in problem.getSuccessors(state):
            new_g = g + cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                h = heuristic(next_state, goal, domain, objects)
                frontier.push((next_state, actions + [action]), new_g + h)

    return []
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
