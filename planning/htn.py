from __future__ import annotations

from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable

class HLA:

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:

        self.name = name

        self.refinements = refinements or []

    def __repr__(self) -> str:

        return f"HLA({self.name})"

def is_primitive(action: Action | HLA) -> bool:

    return isinstance(action, Action)

def is_plan_primitive(plan: list[Action | HLA]) -> bool:

    return all(is_primitive(step) for step in plan)

def execute_primitive_plan(problem: Problem, plan: list[Action]) -> bool:

    state = problem.initial_state

    for action in plan:

        if not is_applicable(state, action):

            return False

        state = apply_action(state, action)

    return problem.isGoalState(state)

def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:

    """

    BFS sobre planes jerárquicos.

    Reemplaza la primera HLA por uno de sus refinamientos hasta obtener

    un plan completamente primitivo y válido.

    """

    frontier = deque()

    frontier.append(hlas)

    visited = set()

    while frontier:

        plan = frontier.popleft()

        problem._expanded += 1

        signature = tuple(str(step) for step in plan)

        if signature in visited:

            continue

        visited.add(signature)

        if is_plan_primitive(plan):

            primitive_plan = [step for step in plan if isinstance(step, Action)]

            if execute_primitive_plan(problem, primitive_plan):

                return primitive_plan

            continue

        first_hla_index = None

        for i, step in enumerate(plan):

            if isinstance(step, HLA):

                first_hla_index = i

                break

        if first_hla_index is None:

            continue

        hla = plan[first_hla_index]

        for refinement in hla.refinements:

            new_plan = (

                plan[:first_hla_index]

                + refinement

                + plan[first_hla_index + 1:]

            )

            frontier.append(new_plan)

    return []

def build_htn_hierarchy(problem: Problem) -> list[HLA]:

    """

    Construye la jerarquía HTN para SimpleRescueProblem y MultiRescueProblem.

    """

    objects = problem.objects

    state = problem.initial_state

    robot = objects["robots"][0]

    patients = objects["patients"]

    supplies = objects["supplies"]

    medical_posts = objects["medical_posts"]

    if not patients or not supplies or not medical_posts:

        return []

    medical_post = medical_posts[0]

    def get_location(obj):

        for fluent in state:

            if len(fluent) == 3 and fluent[0] == "At" and fluent[1] == obj:

                return fluent[2]

        return None

    def make_action(name, precond_pos, precond_neg, add_list, del_list):

        return Action(name, precond_pos, precond_neg, add_list, del_list)

    def move_action(from_cell, to_cell):

        return make_action(

            f"Move({robot}, {from_cell}, {to_cell})",

            [

                ("At", robot, from_cell),

                ("Adjacent", from_cell, to_cell),

                ("Free", to_cell),

            ],

            [],

            [

                ("At", robot, to_cell),

                ("Free", from_cell),

            ],

            [

                ("At", robot, from_cell),

                ("Free", to_cell),

            ],

        )

    def pickup_action(obj, loc):

        return make_action(

            f"PickUp({robot}, {obj}, {loc})",

            [

                ("At", robot, loc),

                ("At", obj, loc),

                ("HandsFree", robot),

                ("Pickable", obj),

            ],

            [],

            [

                ("Holding", robot, obj),

            ],

            [

                ("At", obj, loc),

                ("HandsFree", robot),

            ],

        )

    def putdown_action(obj, loc):

        return make_action(

            f"PutDown({robot}, {obj}, {loc})",

            [

                ("At", robot, loc),

                ("Holding", robot, obj),

            ],

            [],

            [

                ("At", obj, loc),

                ("HandsFree", robot),

            ],

            [

                ("Holding", robot, obj),

            ],

        )

    def rescue_action(patient, loc):

        return make_action(

            f"Rescue({robot}, {patient}, {loc})",

            [

                ("At", robot, loc),

                ("At", patient, loc),

                ("MedicalPost", loc),

                ("SuppliesReady", loc),

            ],

            [],

            [

                ("Rescued", patient),

            ],

            [

                ("At", patient, loc),

            ],

        )

    def setup_supplies_action(supply, loc):

        return make_action(

            f"SetupSupplies({robot}, {supply}, {loc})",

            [

                ("At", robot, loc),

                ("MedicalPost", loc),

                ("Holding", robot, supply),

            ],

            [],

            [

                ("SuppliesReady", loc),

                ("HandsFree", robot),

            ],

            [

                ("Holding", robot, supply),

            ],

        )

    adjacency = {}

    for fluent in state:

        if len(fluent) == 3 and fluent[0] == "Adjacent":

            a = fluent[1]

            b = fluent[2]

            adjacency.setdefault(a, []).append(b)

    def shortest_paths(start, goal, max_paths=3):

        if start == goal:

            return [[]]

        queue = deque()

        queue.append((start, []))

        found_paths = []

        shortest_length = None

        while queue:

            current, path = queue.popleft()

            if shortest_length is not None and len(path) >= shortest_length:

                continue

            for nxt in adjacency.get(current, []):

                if nxt in path:

                    continue

                new_path = path + [nxt]

                if nxt == goal:

                    shortest_length = len(new_path)

                    found_paths.append(new_path)

                    if len(found_paths) >= max_paths:

                        return found_paths

                else:

                    queue.append((nxt, new_path))

        return found_paths

    def navigate_hla(start, goal):

        refinements = []

        for path in shortest_paths(start, goal):

            current = start

            actions = []

            for nxt in path:

                actions.append(move_action(current, nxt))

                current = nxt

            refinements.append(actions)

        return HLA(f"Navigate({start}, {goal})", refinements)

    def prepare_supplies_hla(supply):

        supply_loc = get_location(supply)

        robot_loc = get_location(robot)

        if supply_loc is None or robot_loc is None:

            return HLA(f"PrepareSupplies({supply}, {medical_post})", [])

        return HLA(

            f"PrepareSupplies({supply}, {medical_post})",

            [

                [

                    navigate_hla(robot_loc, supply_loc),

                    pickup_action(supply, supply_loc),

                    navigate_hla(supply_loc, medical_post),

                    setup_supplies_action(supply, medical_post),

                ]

            ],

        )

    def extract_patient_hla(patient):

        patient_loc = get_location(patient)

        if patient_loc is None:

            return HLA(f"ExtractPatient({patient}, {medical_post})", [])

        return HLA(

            f"ExtractPatient({patient}, {medical_post})",

            [

                [

                    navigate_hla(medical_post, patient_loc),

                    pickup_action(patient, patient_loc),

                    navigate_hla(patient_loc, medical_post),

                    putdown_action(patient, medical_post),

                    rescue_action(patient, medical_post),

                ]

            ],

        )

    def full_rescue_mission_hla(patient, supply):

        prepare = prepare_supplies_hla(supply)

        extract = extract_patient_hla(patient)

        return HLA(

            f"FullRescueMission({supply}, {patient}, {medical_post})",

            [

                [prepare, extract],

                [extract],

            ],

        )

    root_tasks = []

    for i, patient in enumerate(patients):

        supply = supplies[min(i, len(supplies) - 1)]

        root_tasks.append(full_rescue_mission_hla(patient, supply))

    return root_tasks