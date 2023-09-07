import copy
import itertools
import search
import numpy

ids = ["212699581", "206465122"]
MOVE_ACTION = "move"
PICK_ACTION = "pick up"
DROP_ACTION = "drop off"
REFUEL_ACTION = "refuel"
WAIT_ACTION = "wait"


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial["map"]
        self.taxis = initial["taxis"]
        self.passengers = initial["passengers"]
        self.gas = []
        self.passenger_dests = [e["destination"] for e in self.passengers.values()]

        self.h_7_i = 0

        self.solvable = True

        I = []
        for y, row in enumerate(self.map):
            for x, col in enumerate(row):
                if col == "I":
                    I.append((y, x))
        flag = False
        for name, data in self.passengers.items():
            if (data["destination"] in I) or (data["location"] in I):
                flag = True
                break
        if flag:
            self.solvable = False

        # self.map_graph =

        for i, row in enumerate(self.map):
            for j, col in enumerate(row):
                if col == "G":
                    self.gas.append((i, j))

        self.problems = initial

        root_state = [[], []]

        for k, v in self.taxis.items():
            root_state[0].append((
                k,
                v["location"],
                v["fuel"],
                ()
            ))
        root_state[0] = tuple(root_state[0])

        for k, v in self.passengers.items():
            root_state[1].append((
                k,
                v["location"],
                None
            ))
        root_state[1] = tuple(root_state[1])
        root_state = tuple(root_state)

        search.Problem.__init__(self, root_state)

    def actions(self, state):
        if not self.solvable:
            return []


        curr_locations = {}
        options_all_taxis = []
        for i in range(len(state[0])):
            curr_locations[state[0][i][0]] = state[0][i][1]
        for i in range(len(state[0])):
            options = []
            for passenger in state[1]:
                if passenger[1] == (state[0][i][1][0], state[0][i][1][1]) and passenger[0] not in state[0][i][3] and len(state[0][i][3]) < \
                        self.problems["taxis"][state[0][i][0]]["capacity"] and not (passenger[1] == self.passengers[passenger[0]]["destination"])\
                        and passenger[2] is None:
                    options.append(("pick up", state[0][i][0], passenger[0]))

            flag = False
            for passenger in state[0][i][3]:
                if self.problems["passengers"][passenger]["destination"] == state[0][i][1]:
                    options.append(("drop off", state[0][i][0], passenger))
                    flag = True

            if (self.problems["map"][state[0][i][1][0]][state[0][i][1][1]] == "G") and (state[0][i][2] != self.taxis[state[0][i][0]]["fuel"]):
                options.append(("refuel", state[0][i][0]))
            if flag:
                options.append(("wait", state[0][i][0]))
                options_all_taxis.append(options)
                continue

            if state[0][i][2] > 0:  # if have gas
                if (state[0][i][1][0] > 0) and self.problems["map"][state[0][i][1][0] - 1][state[0][i][1][1]] != "I":
                    options.append(("move", state[0][i][0], (state[0][i][1][0] - 1, state[0][i][1][1])))  # trying to move up

                if (state[0][i][1][0] < len(self.problems["map"]) - 1) and \
                        self.problems["map"][state[0][i][1][0] + 1][
                            state[0][i][1][1]] != "I":
                    options.append(
                        ("move", state[0][i][0], (state[0][i][1][0] + 1, state[0][i][1][1])))  # move down

                if (state[0][i][1][1] > 0) and self.problems["map"][state[0][i][1][0]][state[0][i][1][1] - 1] != "I":
                    options.append(("move", state[0][i][0], (state[0][i][1][0], state[0][i][1][1] - 1)))  # move left

                if state[0][i][1][1] < len(self.problems["map"][0]) - 1:
                    if self.problems["map"][state[0][i][1][0]][state[0][i][1][1] + 1] != "I":
                        options.append(("move", state[0][i][0], (state[0][i][1][0], state[0][i][1][1] + 1)))  # move right

            options.append(("wait", state[0][i][0]))
            options_all_taxis.append(options)

        if len(options_all_taxis) == 1:
            return ((e, ) for e in options_all_taxis[0])
        combinations = []
        for comb in itertools.product(*options_all_taxis):
            set_locations = set()
            to_locations = set()
            flag = True

            Move_Actions = sum([True for e in comb if e[0] == MOVE_ACTION])

            if Move_Actions > 1:
                for e in comb:
                    if (e[0] == MOVE_ACTION) and (e[-1] in to_locations):
                        flag = False
                        break
                    else:
                        to_locations.add(e[-1])

            for taxi in comb:
                if curr_locations[taxi[1]] in set_locations:
                    flag = False
                    break
                else:
                    set_locations.add(curr_locations[taxi[1]])
            if flag:
                combinations.append(comb)

        return combinations

    def __find(self, name, arr):
        for i, e in enumerate(arr):
            if e[0] == name:
                return i
        return None

    def __execute_action(self, state, action):
        """
        :param state:
        :param action: the atomic action
        :return: returns the execution of the action on the atomic action
        """

        # optimize
        if action[0] == MOVE_ACTION:  # action.type == MOVE_ACTION
            # ("move", "taxi 2", (3, 2))
            taxis = state[0]
            taxi = taxis[self.__find(action[1], state[0])]
            taxi[1] = (action[2][0], action[2][1])  # taxi.location == e.location
            taxi[2] -= 1
        elif action[0] == PICK_ACTION:
            # ("pick up", "taxi 2", "Yossi")
            taxis = state[0]
            passengers = state[1]

            taxi = taxis[self.__find(action[1], state[0])]
            taxi[-1].append(action[2])

            passenger = passengers[self.__find(action[2], state[1])]

            passenger[2] = taxi[0]

        elif action[0] == DROP_ACTION:
            # ("drop off", "taxi 2", "Yossi")
            taxis = state[0]
            passengers = state[1]

            taxi = taxis[self.__find(action[1], state[0])]
            taxi[3].remove(action[2])

            passenger = passengers[self.__find(action[2], state[1])]
            passenger[2] = None
            passenger[1] = (taxi[1][0], taxi[1][1])

        elif action[0] == REFUEL_ACTION:
            # ("refuel", "taxi 2")
            taxis = state[0]
            taxi = taxis[self.__find(action[1], state[0])]
            taxi[2] = self.taxis[taxi[0]]["fuel"]
        return

    def __copy(self, obj):
        cpy = []
        for e in obj:
            cpy.append(e)
        return cpy

    def __get_mutate_state(self, state):
        new_state = [[], []]
        for taxi in state[0]:
            new_state[0].append(
                [
                    taxi[0],
                    (taxi[1][0], taxi[1][1]),
                    taxi[2],
                    self.__copy(taxi[3])
                ]
            )
        for passenger in state[1]:
            new_state[1].append(
                [
                    passenger[0],
                    (passenger[1][0], passenger[1][1]),
                    passenger[2]
                ]
            )
        return new_state

    def __get_immu_state(self, new_state):
        for i, taxi in enumerate(new_state[0]):
            taxi[3] = tuple(taxi[3])
            new_state[0][i] = tuple(taxi)
        new_state[0] = tuple(new_state[0])

        for i, passenger in enumerate(new_state[1]):
            new_state[1][i] = tuple(passenger)
        new_state[1] = tuple(new_state[1])
        return tuple(new_state)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = self.__get_mutate_state(state)

        for e in action:  # e is atomic action
            self.__execute_action(new_state, e)

        return self.__get_immu_state(new_state)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

        for e in state[1]:
            if e[2] is not None:
                return False

            if self.man_distance(e[1], self.passengers[e[0]]["destination"]) != 0:
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        if not self.solvable:
            return 0

        # print("-" * 10,
        #     "self.h_1(node): " + str(self.h_1(node)),
        #     "self.h_2(node): " + str(self.h_2(node)),
        #     "self.h_3(node): " + str(self.h_3(node)),
        #     "self.h_4(node): " + str(self.h_4(node)),
        #     "self.h_5(node): " + str(self.h_5(node)),
        #     "self.h_6(node): " + str(self.h_6(node)),
        #     "self.h_7(node): " + str(self.h_7(node)),
        #     "self.h_8(node): " + str(self.h_8(node)),
        #     "self.h_9(node): " + str(self.h_9(node)),
        #     sep='\n',
        #     end='\r')

        # print("Running")

        return max(
            0,
            # self.h_1(node),
            # self.h_2(node) * len(node.state[0]) / sum([e["capacity"] for e in self.taxis.values()]),
            # self.h_3(node),  # alone 3.1s
            # self.h_4(node),
            # self.h_5(node),
            # self.h_6(node),
            self.h_7(node),
            # self.h_8(node),
            # self.h_9(node),
        )

    def h_1(self, node):
        picked_passengers = 0
        for taxi in node.state[0]:
            picked_passengers += len(taxi[3])
        unpicked_passengers = 0
        for passenger in node.state[1]:
            if passenger[1] != self.problems["passengers"][passenger[0]]["destination"]:
                unpicked_passengers += 1
        unpicked_passengers = unpicked_passengers - picked_passengers
        return (unpicked_passengers * 2 + picked_passengers)/len(node.state[0])

    def calc_sum_man_distance(self, state):
        A = []
        for e in state[1]:
            if e[2] is None:
                p1 = e[1]
            else:
                p1 = state[0][self.__find(e[2], state[0])][1]

            A.append(
                self.man_distance(p1, self.passengers[e[0]]["destination"])
            )

        return sum(A)

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        return self.calc_sum_man_distance(node.state)/len(node.state[0])

    def man_distance(self, p1, p2):
        return sum([abs(e1 - e2) for e1, e2 in zip(p1, p2)])

    def h_3(self, node):
        p_d = []
        t_p = []
        for passenger in node.state[1]:
            p_loc = passenger[1] if passenger[2] is None else node.state[0][self.__find(passenger[2], node.state[0])][1]
            p_dest = self.passengers[passenger[0]]["destination"]

            p_d.append(self.dist(p_loc, p_dest))
            if passenger[2] is None:
                taxi_locations = []
                for taxi in node.state[0]:
                    taxi_locations.append((taxi[1][0], taxi[1][1]))
                t_p.append(self.dist(p_loc, taxi_locations))
            else:
                t_p.append(0)

        return sum([e1 + e2 for e1, e2 in zip(t_p, p_d)])

    def h_4(self, node):
        p_d = []
        for passenger in node.state[1]:
            p_loc = passenger[1] if passenger[2] is None else node.state[0][self.__find(passenger[2], node.state[0])][1]
            p_d.append(self.man_distance(p_loc, self.passengers[passenger[0]]["destination"]))
        return sum(p_d)

    def h_5(self, node):
        t_p = []
        for passenger in node.state[1]:
            if passenger[2] is None:
                t_p_temp = []
                for taxi in node.state[0]:
                    t_p_temp.append(self.man_distance(taxi[1], passenger[1]))
                t_p.append(min(t_p_temp))
            else:
                t_p.append(0)
        return sum(t_p)

    def h_6_(self, state):
        taxis = {}
        # new_state = self.__get_mutate_state(state)

        for taxi in state[0]:
            taxis[taxi[0]] = 0
            min_dist = float('inf')
            p_loc = None
            p_name = None
            for passenger in state[1]:
                if (passenger[2] is not None) and (taxi[0] != passenger[2]):
                    continue
                p_d = self.man_distance(taxi[1], passenger[1])
                if p_d < min_dist:
                    min_dist = p_d
                    p_loc = passenger[1]
                    p_name = passenger[0]

            taxis[taxi[0]] = taxis[taxi[0]] + min_dist + 2 + self.man_distance(p_loc, self.passengers[p_name]["destination"])
        return sum([e for e in taxis.values()])

    def h_6(self, node):
        return self.h_6_(node.state)

    def dist(self, start, end, flag=False):
        if (start is None) or (end is None) or (len(end) == 0):
            return -1

        prob = mapProblem(
            {
                "location": start,
                "map": self.map,
            },
            end
        )
        s = search.breadth_first_search(prob)
        solution = list(map(lambda n: n.action, s.path()))[1:]
        d = len(solution)
        if flag:
            return d, solution[-1]
        return d

    def min_route_through_all(self, points: list, start: int):
        if len(points) <= 1:
            return 0
        s = points.pop(start)

        d, arg = self.dist(s, points, True)
        return d + self.min_route_through_all(points, points.index(arg))

    def priority(self, p_name, passengers, start_location, end_location):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        num_i = 0
        for e in directions:
            if (0 <= end_location[0] + e[0] < len(self.map)) \
                    and (0 <= end_location[1] + e[1] < len(self.map[0])):
                num_i = num_i + (self.map[end_location[0] + e[0]][end_location[1] + e[1]] == 'I')

        prio = 4 - num_i
        d = self.dist(passengers[p_name]['location'], self.passengers[p_name]["destination"])
        prio = prio + (1 / (d if d > 0 else 0.5))
        d = self.dist(self.passengers[p_name]["destination"], end_location)
        prio = prio + (1 / (d if d > 0 else 0.5))
        d = self.dist(self.passengers[p_name]["location"], start_location)
        prio = prio + (1 / (d if d > 0 else 0.25))

        return prio

    def not_goal(self, passengers):
        for p_name, p_dict in passengers.items():
            if (self.passengers[p_name]['destination'] != p_dict['location']) or (p_dict['picked by'] is not None):
                return True
        return False

    def h_7(self, node):
        state = node.state
        passengers = {}
        for e in state[1]:
            p_loc = e[1] if e[2] is None else state[0][self.__find(e[2], state[0])][1]
            passengers[e[0]] = {'location': p_loc,
                                'picked by': e[-1]}
        taxis = {
            e[0]:
                {
                    "location": e[1],
                    "cost": 0,
                    "passengers": [p for p in e[-1]],
                    "space": self.taxis[e[0]]["capacity"] - len([p for p in e[-1]])
                }
            for e in state[0]
        }

        while self.not_goal(passengers):
            p_t = {}
            for p_name, p_dict in passengers.items():
                p_t[p_name] = self.dist(p_dict['location'], [e["location"] for e in taxis.values()])

            t_p = {}
            p_drop = set()
            temp_passengers = copy.deepcopy(passengers)

            for p_name, p_dict in temp_passengers.items():
                if (p_dict['location'] == self.passengers[p_name]['destination']) and (p_dict['picked by'] is None):
                    p_drop.add(p_name)
            for p_name in p_drop:
                temp_passengers.pop(p_name)
            for taxi_name in taxis:
                # if there's space in taxi check for more passengers
                # calc for unpicked passengers
                seen = set()
                t_p[taxi_name] = []

                if taxis[taxi_name]["space"] > 0:
                    p_lookup = []
                    for e in temp_passengers.values():
                        if e['picked by'] is None:
                            p_lookup.append(e['location'])
                    min_t_p = self.dist(taxis[taxi_name]['location'], p_lookup)
                    for p_name, p_dict in temp_passengers.items():
                        dist_t_p = self.dist(taxis[taxi_name]['location'], p_dict['location'])
                        #  this taxi is closest        this passenger is
                        #  to this passenger           closest to this taxi
                        if (dist_t_p == p_t[p_name]) and (dist_t_p == min_t_p) and (passengers[p_name]['picked by'] is None):
                            t_p[taxi_name].append([p_name, passengers[p_name]['location'],
                                                   self.priority(p_name, passengers, taxis[taxi_name]['location'],
                                                                 passengers[p_name]['location'])])
                            seen.add(p_name)
                    for seen_e in seen:
                        temp_passengers.pop(seen_e)
                if len(taxis[taxi_name]["passengers"]) > 0:
                    min_t_d = self.dist(taxis[taxi_name]['location'],
                                        [self.passengers[e]['destination'] for e in taxis[taxi_name]["passengers"]])
                    for p_name in taxis[taxi_name]['passengers']:
                        # p_dict = passengers[p_name]
                        dist_t_d = self.dist(taxis[taxi_name]['location'], self.passengers[p_name]['destination'])
                        if dist_t_d == min_t_d:
                            t_p[taxi_name].append([p_name, self.passengers[p_name]['destination'],
                                                   self.priority(p_name, passengers, taxis[taxi_name]['location'],
                                                                 self.passengers[p_name]['destination'])])

                t_p[taxi_name] = sorted(t_p[taxi_name], key=lambda x: x[-1], reverse=True)

                # if there's no more space in taxi start dropping off

            # move each taxi into the passenger with the highest priority
            for taxi, prio_q in t_p.items():
                if len(prio_q) == 0:
                    continue
                d = self.dist(taxis[taxi]["location"], prio_q[0][1])
                taxis[taxi]["location"] = prio_q[0][1]
                for p in taxis[taxi]['passengers']:
                    passengers[p]['location'] = (taxis[taxi]["location"][0], taxis[taxi]["location"][1])
                taxis[taxi]["cost"] = taxis[taxi]["cost"] + d

                if (taxis[taxi]['space'] == 0) or (prio_q[0][0] in taxis[taxi]['passengers']):
                    continue
                taxis[taxi]["passengers"].append(prio_q[0][0])
                passengers[prio_q[0][0]]['picked by'] = taxi
                taxis[taxi]['space'] = taxis[taxi]['space'] - 1
                taxis[taxi]["cost"] = taxis[taxi]["cost"] + 1

            # drop off passengers if they at destination
            for p_name, p_dict in passengers.items():
                if (p_dict['location'] == self.passengers[p_name]['destination']) and (p_dict['picked by'] is not None):
                    taxis[p_dict['picked by']]['passengers'].remove(p_name)
                    taxis[p_dict['picked by']]['space'] = taxis[p_dict['picked by']]['space'] + 1
                    taxis[p_dict['picked by']]['cost'] = taxis[p_dict['picked by']]['cost'] + 1
                    p_dict['picked by'] = None

        return max([e['cost'] for e in taxis.values()])

    def h_8(self, node):
        passenger_locations = []
        taxi_locations = []
        state = node.state
        passengers_dists = {}

        for taxi in state[0]:
            if len(taxi[3]) < self.taxis[taxi[0]]["capacity"]:
                taxi_locations.append(taxi[1])

        for passenger in state[1]:
            p_loc = passenger[1] if passenger[2] is None else node.state[0][self.__find(passenger[2], node.state[0])][1]
            passenger_locations.append((p_loc[0], p_loc[1]))
            passengers_dists[passenger[0]] = self.dist(p_loc, taxi_locations) if len(taxi_locations) > 0 else 0

        max_pass_val = max(passengers_dists.values())

        m = 0
        for k, v in passengers_dists.items():
            if v == max_pass_val:
                t = self.dist(self.passengers[k]["location"], self.passengers[k]["destination"])
                if t > m:
                    m = t

        return m + max_pass_val

    def h_9(self, node):
        state = node.state
        c = []
        for taxi in state[0]:
            c.append(taxi[2])
        return max(c)




class mapProblem(search.Problem):
    def __init__(self, initial, goal=None):
        self.goal = goal
        self.map = initial["map"]
        self.found_passenger = set()
        search.Problem.__init__(self, initial["location"], goal)

    def actions(self, state):
        ret = []
        row = state[0]
        col = state[1]
        if (row > 0) and (self.map[row - 1][col] != 'I'):
            ret.append((row - 1, col))
        if (col > 0) and (self.map[row][col - 1] != 'I'):
            ret.append((row, col - 1))
        if (row < len(self.map) - 1) and (self.map[row + 1][col] != 'I'):
            ret.append((row + 1, col))
        if (col < len(self.map[row]) - 1) and (self.map[row][col + 1] != 'I'):
            ret.append((row, col + 1))
        return tuple(ret)

    def result(self, state, action):
        return action

    def goal_test(self, state):
        if isinstance(self.goal, list):
            # 1 - taxi to passenger
            return state in self.goal
        else:
            # 2 - passenger to destination
            return state == self.goal





def create_taxi_problem(game):
    return TaxiProblem(game)


# GasStations: []
# Is: []

# state : (
#           taxis: (
#               (name: str, location: (x, y), fuel: int, ('passenger1', ... ): tuple),
#               (name: str, location: (x, y), fuel: int, ('passenger1', ... ): tuple),
#           ),
#           passengers: (
#               (name: str, init_location: (x, y), in_taxi: str),
#               (name: str, init_location: (x, y), in_taxi: str),
#           )
#         )

