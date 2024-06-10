import heapq

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    constraint_table = {}
    for constraint in constraints:
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            if timestep not in constraint_table:
                constraint_table[timestep] = []
            constraint_table[timestep].append(constraint)
    # Return an empty dictionary if constraint_table is None
    return constraint_table if constraint_table else {}


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            if constraint['loc'] == [next_loc] or (
                    len(constraint['loc']) == 2 and constraint['loc'] == [curr_loc, next_loc]):
                print("constraint found in ", next_loc, " at ", next_time)
                return True
    return False

def is_constrained_future(curr_loc, next_loc, next_time, constraint_table, my_map):

    for curr_time in range(next_time, len(my_map[0])*len(my_map[1])):
        if curr_time in constraint_table:
            for constraint in constraint_table[curr_time]:
                if constraint['loc'] == [next_loc] or (
                        len(constraint['loc']) == 2 and constraint['loc'] == [curr_loc, next_loc]):
                    print("constraint found in ", next_loc, " at ", curr_time)
                    return True
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    constraint_table = build_constraint_table(constraints, agent)
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': earliest_goal_timestep}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and not is_constrained_future(curr['loc'], goal_loc,
                                             curr['timestep']+1, constraint_table, my_map):
            return get_path(curr)
        # Generate child nodes for moving to neighboring cells
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep']+1}
            if not is_constrained(curr['loc'], child['loc'], child['timestep'], constraint_table):
                if (child['loc'], child['timestep']) in closed_list:
                    existing_node = closed_list[(child['loc'], child['timestep'])]
                    if compare_nodes(child, existing_node):
                        closed_list[(child['loc'], child['timestep'])] = child
                        push_node(open_list, child)
                else:
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)

        # Generate child node where the agent waits in the current cell
        wait_child = {'loc': curr['loc'],
                      'g_val': curr['g_val'] + 1,
                      'h_val': curr['h_val'],
                      'parent': curr,
                      'timestep': curr['timestep'] + 1}
        if not is_constrained(curr['loc'], wait_child['loc'], wait_child['timestep'], constraint_table):
            if (wait_child['loc'], wait_child['timestep']) in closed_list:
                existing_node = closed_list[(wait_child['loc'], wait_child['timestep'])]
                if compare_nodes(wait_child, existing_node):
                    closed_list[(wait_child['loc'], wait_child['timestep'])] = wait_child
                    push_node(open_list, wait_child)
            else:
                closed_list[(wait_child['loc'], wait_child['timestep'])] = wait_child
                push_node(open_list, wait_child)

    return None  # Failed to find solutions
