import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def get_path_sol(self, goal_node):
        """Extracts the path from the given goal node back to the start node."""
        path = []
        curr = goal_node
        while curr is not None:
            path.append(curr['loc'])
            curr = curr['parent']
        path.reverse()
        return path

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []
        max_timestep = len(self.my_map) * len(self.my_map[0])


        # Add the specific constraint for agent 1
        constraints.append({'agent': 1, 'loc': [(1, 2)], 'timestep': 2})
        constraints.append({'agent': 1, 'loc': [(1, 4)], 'timestep': 2})

        # Find path for each agent
        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints, time_limit=max_timestep)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            # Add constraints for the future agents to avoid collisions
            # First loop: Iterate over the path of the current agent
            for t in range(len(path)):
                # Second loop: Add constraints for all future agents
                for k in range(i + 1, self.num_of_agents):
                    # Add vertex constraint
                    constraints.append({'agent': k, 'loc': [path[t]], 'timestep': t})
                    # Add edge constraint if not the last step in the path
                    if t < len(path) - 1:
                        constraints.append({'agent': k, 'loc': [path[t], path[t + 1]], 'timestep': t + 1})

            # Add constraints for future agents to prevent moving on top of agents that have reached their goal
            for k in range(i + 1, self.num_of_agents):
                for t in range(len(path), max_timestep):
                    constraints.append({'agent': k, 'loc': [path[-1]], 'timestep': t})

            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
