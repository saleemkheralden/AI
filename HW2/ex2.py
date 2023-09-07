import copy
import itertools
import math
import random
import numpy as np

import utils
import time

ids = ["212699581", "206465122"]


class OptimalTaxiAgent:
    def __init__(self, initial, flag=True):
        self.initial = initial
        self.start = copy.deepcopy(initial)
        self.taxis = [[i, self.initial["taxis"][i]["location"], [], self.initial["taxis"][i]["fuel"]] for i in
                      self.initial["taxis"].keys()]
        self.passengers = [
            [i, "no taxi", self.initial["passengers"][i]["location"], self.initial["passengers"][i]["destination"]] for
            i in self.initial["passengers"].keys()]
        self.initial_state = [self.taxis, self.passengers, self.initial["turns to go"]]
        self.val_func = {}
        self.taxi_to_index = {taxi: idx for idx, taxi in enumerate(self.initial["taxis"].keys())}
        self.index_to_passenger = {idx: passenger for idx, passenger in enumerate(self.initial["passengers"].keys())}
        self.passenger_to_index = {passenger: idx for idx, passenger in enumerate(self.initial["passengers"].keys())}

        if flag:
            self.val_fun(self.initial_state)

    def array_to_tuple(self, state):
        taxis = copy.deepcopy(state[0])
        for i in range(len(taxis)):
            taxis[i][2] = tuple(taxis[i][2])
            taxis[i] = tuple(taxis[i])
        taxis = tuple(taxis)
        passengers = copy.deepcopy(state[1])
        for i in range(len(passengers)):
            passengers[i] = tuple(passengers[i])
        passengers = tuple(passengers)
        t_state = (taxis, passengers, state[2])
        return t_state

    def val_fun(self, new_state):
        max_val = 0
        best_act = ("terminate",)
        state = copy.deepcopy(new_state)
        state_t1 = self.array_to_tuple(state)

        destinations = self.possible_destinations(state)
        states = self.possible_states(state)
        for act in states:
            if act[0][0] == "terminate":
                continue
            val = 0
            state_curr = copy.deepcopy(act[1])
            if act[0][0] == "reset":
                state_t = self.array_to_tuple(state_curr)
                reward = self.cal_sum(act[0])
                if state_t[2] == 0:
                    val += reward
                else:
                    if state_t not in self.val_func.keys():
                        self.val_fun(state_curr)
                    val += (self.val_func[state_t][1] + reward)
                if val >= max_val:
                    max_val = val
                    best_act = act[0]
                continue
            for dest in destinations:
                for i in range(len(dest[1])):
                    state_curr[1][i][3] = dest[1][i]
                state_t = self.array_to_tuple(state_curr)
                reward = self.cal_sum(act[0])
                if state_t[2] == 0:
                    val += dest[0] * reward
                else:
                    if state_t not in self.val_func.keys():
                        self.val_fun(state_curr)
                    val += dest[0] * (self.val_func[state_t][1] + reward)
            if val >= max_val:
                max_val = val
                best_act = act[0]
        self.val_func[state_t1] = (best_act, max_val)
        return

    def cal_sum(self, act):
        sum1 = 0
        for i in act:
            if i[0] == "drop off":
                sum1 += 100
            if i == "reset":
                sum1 -= 50
            if i[0] == "refuel":
                sum1 -= 10
        return sum1

    def possible_destinations(self, state, passenger_keys=None):
        location_combinations = []
        if passenger_keys is None:
            passenger_keys = self.initial["passengers"].keys()
        for passenger in passenger_keys:
            poss_locations = [self.initial["passengers"][passenger]["destination"]]
            for loc in self.initial["passengers"][passenger]["possible_goals"]:
                poss_locations.append(loc)
            location_combinations.append(poss_locations)
        options = set()
        for dest in itertools.product(*location_combinations):
            prob = 1
            for i in range(len(dest)):
                pass_prob = self.initial["passengers"][self.index_to_passenger[i]]["prob_change_goal"]
                if len(self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"]) != 1:
                    if dest[i] == state[1][i][3]:
                        if dest[i] in self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"]:
                            prob *= ((1 - pass_prob) + (pass_prob / (len(self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"]))))
                        else:
                            prob *= (1 - pass_prob)
                    else:
                        prob *= (pass_prob / (len(self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"])))
            options.add((prob, dest))
        return options

    def possible_states(self, state):
        actions = self.possible_actions(state)
        states = []
        for action in actions:
            if action[0] == "terminate":
                states.append((action, ()))
                continue
            if action[0] == "reset":
                curr_state = copy.deepcopy(self.initial_state)
                curr_state[2] = state[2] - 1
                states.append((action, curr_state))
                continue
            curr_state = copy.deepcopy(state)
            curr_state[2] = state[2] - 1
            for task in action:
                if task[0] == "refuel":
                    curr_state[0][self.taxi_to_index[task[1]]][3] = self.initial["taxis"][task[1]]["fuel"]
                if task[0] == "pick up":
                    curr_state[0][self.taxi_to_index[task[1]]][2].append(task[2])
                    curr_state[1][self.passenger_to_index[task[2]]][1] = task[1]
                    curr_state[1][self.passenger_to_index[task[2]]][2] = task[1]
                if task[0] == "drop off":
                    curr_state[0][self.taxi_to_index[task[1]]][2].remove(task[2])
                    curr_state[1][self.passenger_to_index[task[2]]][1] = "no taxi"
                    curr_state[1][self.passenger_to_index[task[2]]][2] = curr_state[0][self.taxi_to_index[task[1]]][1]
                if task[0] == "move":
                    curr_state[0][self.taxi_to_index[task[1]]][1] = task[2]
                    curr_state[0][self.taxi_to_index[task[1]]][3] -= 1
            states.append((action, curr_state))
        return states

    def possible_actions(self, state):
        actions_per_taxis = []

        for taxi in state[0]:
            curr_states = [("wait", taxi[0])]

            if self.initial["map"][taxi[1][0]][taxi[1][1]] == "G":
                curr_states.append(("refuel", taxi[0]))

            for p in state[1]:
                if len(taxi[2]) < self.initial["taxis"][taxi[0]]["capacity"] and p[1] == "no taxi" and p[2] == taxi[1] and p[2] != p[3]:
                    curr_states.append(("pick up", taxi[0], p[0]))

            for p in taxi[2]:
                if taxi[1] == state[1][self.passenger_to_index[p]][3]:
                    curr_states.append(("drop off", taxi[0], p))

            if taxi[1][0] > 0 and self.initial["map"][taxi[1][0] - 1][taxi[1][1]] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0] - 1, taxi[1][1])))

            if taxi[1][0] < len(self.initial["map"]) - 1 and self.initial["map"][taxi[1][0] + 1][taxi[1][1]] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0] + 1, taxi[1][1])))

            if taxi[1][1] > 0 and self.initial["map"][taxi[1][0]][taxi[1][1] - 1] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0], taxi[1][1] - 1)))

            if taxi[1][1] < len(self.initial["map"][0]) - 1 and self.initial["map"][taxi[1][0]][taxi[1][1] + 1] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0], taxi[1][1] + 1)))

            actions_per_taxis.append(curr_states)

        combinations = []
        for comb in itertools.product(*actions_per_taxis):
            curr_locations = []
            for i in range(len(state[0])):
                curr_locations.append(state[0][i][1])
            for taxi in comb:
                if taxi[0] == "move":
                    curr_locations.remove(state[0][self.taxi_to_index[taxi[1]]][1])
                    curr_locations.append(taxi[2])
            if len(set(curr_locations)) == len(curr_locations):
                combinations.append(comb)
        combinations.append(("terminate",))
        combinations.append(("reset",))

        return combinations

    def act(self, state, flag=True):
        if flag:
            taxis_pass = {}
            for taxi in state["taxis"].keys():
                taxis_pass[taxi] = []
            for pas in state["passengers"].keys():
                if type(state["passengers"][pas]["location"]) == str:
                    taxis_pass[state["passengers"][pas]["location"]].append(pas)
            taxis = tuple(
                [tuple([i, state["taxis"][i]["location"], tuple(taxis_pass[i]), state["taxis"][i]["fuel"]]) for i in
                 state["taxis"].keys()])
            passengers = tuple(
                [tuple([i, state["passengers"][i]["location"] if type(
                    state["passengers"][i]["location"]) == str else "no taxi", state["passengers"][i]["location"],
                        state["passengers"][i]["destination"]]) for i in
                 state["passengers"].keys()])
            state = (taxis, passengers, state["turns to go"])
        x = self.val_func[state][0]
        if x[0] == "reset" or x[0] == "terminate":
            return x[0]
        return x


from threading import Thread


class thread(Thread):
    def __init__(self):
        super().__init__()

    def update_func(self, func):
        self.func = func

    def update_params(self, params):
        self.params = params

    def run(self):
        self.func(self.params)


class TaxiAgent(OptimalTaxiAgent):
    def __init__(self, initial):
        # print("starting __init__")
        # start = time.perf_counter()
        super().__init__(initial, flag=False)
        self.initial = copy.deepcopy(initial)
        self.map = self.initial['map']

        self.locations_of_stations = []
        for i, row in enumerate(initial["map"]):
            for j, col in enumerate(row):
                if col == "G":
                    self.locations_of_stations.append((i, j))

        self.cumulative_points = 0

        self.Q_ = Q(taxi_agent=self, default=True)
        self.q_learning = Q_learning(self.Q_, self)

        # self.q_learning.train(self.initial_state, timeout=10)
        # print(self.Q_.theta)
        # print("finished __init__")

    def act(self, state, flag=True):
        start = time.perf_counter()
        taxis_pass = {}
        for taxi in state["taxis"].keys():
            taxis_pass[taxi] = []
        for pas in state["passengers"].keys():
            if type(state["passengers"][pas]["location"]) == str:
                taxis_pass[state["passengers"][pas]["location"]].append(pas)
        taxis = [[i, state["taxis"][i]["location"], taxis_pass[i], state["taxis"][i]["fuel"]]
                 for i in state["taxis"].keys()]
        passengers = [[i,
                       state["passengers"][i]["location"] if type(state["passengers"][i]["location"]) == str
                       else "no taxi",
                       state["passengers"][i]["location"],
                       state["passengers"][i]["destination"]] for i in
                      list(state["passengers"].keys())[:len(taxis)]]
        state_arr = [taxis, passengers, state["turns to go"]]

        pos_actions = self.possible_actions(state_arr)

        maxq = self.Q_.calc(state_arr, pos_actions[0])
        maxAction = pos_actions[0]

        # print('*' * 15)
        # print(maxAction, maxq)

        for action in pos_actions[1:]:
            if time.perf_counter() - start > 0.08:
                return maxAction

            tempCalc = self.Q_.calc(state_arr, action)
            # print(action, '\t', tempCalc)
            if tempCalc > maxq:
                maxq = tempCalc
                maxAction = action
        # print('*' * 15)

        # print(maxAction)
        # print(state['taxis'])
        # print(state['passengers'])
        # print(state['turns to go'])

        self.cumulative_points = self.cumulative_points + self.cal_sum(maxAction)

        if maxAction[0] == "reset" or maxAction[0] == "terminate":
            return maxAction[0]

        return maxAction

    def exec_action(self, state, action):
        if action[0] == "terminate":
            return ()
        if action[0] == "reset":
            curr_state = copy.deepcopy(self.initial_state)
            curr_state[2] = state[2] - 1
            return curr_state

        curr_state = copy.deepcopy(state)
        curr_state[2] = state[2] - 1

        for task in action:
            if task[0] == "refuel":
                curr_state[0][self.taxi_to_index[task[1]]][3] = self.initial["taxis"][task[1]]["fuel"]
            if task[0] == "pick up":
                curr_state[0][self.taxi_to_index[task[1]]][2].append(task[2])
                curr_state[1][self.passenger_to_index[task[2]]][1] = task[1]
                curr_state[1][self.passenger_to_index[task[2]]][2] = task[1]
            if task[0] == "drop off":
                curr_state[0][self.taxi_to_index[task[1]]][2].remove(task[2])
                curr_state[1][self.passenger_to_index[task[2]]][1] = "no taxi"
                curr_state[1][self.passenger_to_index[task[2]]][2] = curr_state[0][self.taxi_to_index[task[1]]][1]
            if task[0] == "move":
                curr_state[0][self.taxi_to_index[task[1]]][1] = task[2]
                curr_state[0][self.taxi_to_index[task[1]]][3] -= 1
        return curr_state


class Q:
    def __init__(self, taxi_agent: TaxiAgent, lr=.6, gamma=.7, default=False):
        self.features = [lambda *args: 1, self.taxi_locations,
                         self.passenger_locations, self.fuel_levels]

        if default:
            self.theta = np.array([0.32063316, 0.26173338,
                                   0.12682486, 0.2908086])
        else:
            self.theta = np.random.normal(1, 1, len(self.features))

        self.taxi_agent = taxi_agent
        self.lr = lr
        self.gamma = gamma

    def calc(self, state, action):
        if len(state) == 0:
            return 0
        possible_destinations = self.taxi_agent.possible_destinations(state, passenger_keys=[e[0] for e in state[1]])

        # f1 = self.features[0](state, action, possible_destinations)
        # f2 = self.features[1](state, action, possible_destinations)
        # f3 = self.features[2](state, action, possible_destinations)
        # f4 = self.features[3](state, action, possible_destinations)
        #
        # return self.theta[0] * f1 + self.theta[1] * f2 + self.theta[2] * f3 + self.theta[3] * f4

        return sum([theta_i * f(state, action, possible_destinations)
                    for theta_i, f in zip(self.theta, self.features)])

    def maxQ(self, state, flag=False):
        if len(state) == 0:
            return 0

        actions = self.taxi_agent.possible_actions(state)
        maxq = self.calc(state, actions[0])
        maxAction = actions[0]

        for action in actions[1:]:
            tempCalc = self.calc(state, action)
            if tempCalc > maxq:
                maxq = tempCalc
                maxAction = action

        if flag:
            return maxq, maxAction

        return maxq

    def train(self, s, s_prime, a):
        possible_destinations = self.taxi_agent.possible_destinations(s)
        for i, theta_i in enumerate(self.theta):

            fi = self.features[i](s, a, possible_destinations)
            selfcalc = self.calc(s, a)
            selfmax = self.maxQ(s_prime)
            reward_a = self.reward(a)
            sumall = (self.reward(a) + self.gamma * self.maxQ(s_prime) - self.calc(s, a))
            lrXsumallXfi = self.lr * sumall * fi

            self.theta[i] = theta_i + self.lr * \
                (self.reward(a) + self.gamma * self.maxQ(s_prime) - self.calc(s, a)) * \
                self.features[i](s, a, possible_destinations)

        sum_theta = sum([abs(e) for e in self.theta])
        # sum_theta = sum(self.theta)

        for i, theta_i in enumerate(self.theta):
            self.theta[i] = abs(theta_i / sum_theta)
            # self.theta[i] = theta_i / sum_theta

    def reward(self, a):
        return self.taxi_agent.cal_sum(a)

    def closest_passenger(self, corr, passengers, index=2, execlude=[], flag=False):
        min_dist = float('inf')
        min_passenger = None
        for passenger in passengers:
            if (index == 2) and ((passenger[1] != 'no taxi') or (passenger[0] in execlude)):
                continue

            # dist = self.man_distance(corr, passenger[index])
            dist = self.BFS_distance(corr, passenger[index])
            if dist < min_dist:
                min_dist = dist
                min_passenger = passenger

        if flag:
            min_passengers = []
            for passenger in passengers:
                if (passenger[1] != 'no taxi') or (passenger[0] in execlude):
                    continue
                if self.BFS_distance(corr, passenger[index]) == min_dist:
                    min_passengers.append(passenger)
            return min_passengers, min_dist

        return min_passenger, min_dist

    def closest_taxi(self, corr, taxis, index=1, execlude=[], flag=False):
        min_dist = float('inf')
        min_taxi = None
        for taxi in taxis:
            if taxi[0] in execlude:
                continue

            dist = self.BFS_distance(corr, taxi[index])
            if dist < min_dist:
                min_dist = dist
                min_taxi = taxi

        if flag:
            min_taxis = []
            for taxi in taxis:
                if taxi[0] in execlude:
                    continue
                if self.BFS_distance(corr, taxi[index]) == min_dist:
                    min_taxis.append(taxi)
            return min_taxis, min_dist

        return min_taxi, min_dist

    def calc_action(self, state, action, possible_destinations):
        points = 0
        execlude = []

        if action[0] == "reset":
            points -= 150
            return points
        elif action[0] == "terminate":
            for taxi in state[0]:
                if (taxi[3] == 0) and (taxi[1] not in self.taxi_agent.locations_of_stations):
                    points += 50
                    return points
            if self.taxi_agent.cumulative_points <= 0:
                points -= 100
            return points

        for taxi in state[0]:
            # get the corresponding action to the taxi
            taxi_action = None
            for atom_action in action:
                if atom_action[1] == taxi[0]:
                    taxi_action = atom_action
                    break

            if taxi_action is None:
                continue

            if taxi_action[0] == "move":
                closest_passengers, passenger_dist = \
                    self.closest_passenger(taxi[1], state[1], execlude=execlude, flag=True)

                points += 1

                if len(taxi[2]) > 0:
                    closest_dest, dest_dist = self.closest_passenger(taxi[1], state[1], 3)
                    if dest_dist < passenger_dist:
                        if self.BFS_distance(taxi_action[2], closest_dest[3]) < \
                                self.BFS_distance(taxi[1], closest_dest[3]):
                            points += 10

                if len(closest_passengers) > 0:
                    for passenger in closest_passengers:
                        aaaa = self.closest_taxi(passenger[2], state[0])
                        if self.closest_taxi(passenger[2], state[0])[1] == passenger_dist:
                            if self.BFS_distance(taxi_action[2], passenger[2]) < \
                                    self.BFS_distance(taxi[1], passenger[2]):
                                points += 5
                            execlude.append(passenger)
                            break

                # for dest in possible_destinations:
                #     prob = dest[0]
                #     for i, passenger_dest in enumerate(dest[1]):
                #         curr_state[1][i][3] = passenger_dest
                #     if len(taxi[2]) > 0:
                #         closest_dest, dest_dist = self.closest_passenger(taxi[1], curr_state[1], 3)
                #         if dest_dist < passenger_dist:
                #             if self.man_distance(taxi_action[2], closest_dest[3]) < \
                #                     self.man_distance(taxi[1], closest_dest[3]):
                #                 points += prob * 10
                #                 continue
                #     if closest_passenger is not None:
                #         if self.man_distance(taxi_action[2], closest_passenger[2]) < \
                #                 self.man_distance(taxi[1], closest_passenger[2]):
                #             points += prob * 10
            elif taxi_action[0] == "drop off":
                points += 100
            elif taxi_action[0] == "pick up":
                points += 50
            # elif taxi_action[0] == "refuel":
            #     points += -10
            elif taxi_action[0] == "wait":
                points -= 1
            else:
                points += 0

        return points

    def calc_utility(self, state):
        if len(state) == 0:
            return 0

        locations_of_taxis = []
        # taxis_assignment = {}
        utility = 0
        for taxi in state[0]:
            locations_of_taxis.append(taxi[1])
            # taxis_assignment[taxi[0]] = []

        for passenger in state[1]:
            if passenger[1] != 'no taxi':
                continue

            # closest_taxi, closest_dest = self.closest_passenger(passenger[2], state[0], index=1)
            # taxis_assignment[closest_taxi[0]].append((passenger, closest_dest))

            closest_taxi = self.BFS_distance(passenger[2], locations_of_taxis[0])
            for taxi in locations_of_taxis:
                curr_dist = self.BFS_distance(taxi, passenger[2])
                if curr_dist < closest_taxi:
                    closest_taxi = curr_dist
            utility += closest_taxi**1.2
        return utility

    # f_1(s, a)
    def taxi_locations(self, state, action, possible_destinations):
        return self.calc_action(state, action, possible_destinations)

    # f_2(s, a)
    def passenger_locations(self, state, action, possible_destinations):
        new_state = self.taxi_agent.exec_action(state, action)
        return self.calc_utility(state) - self.calc_utility(new_state)

    # f_3(s, a)
    def fuel_levels(self, state, action, possible_destinations):
        if len(self.taxi_agent.locations_of_stations) == 0:
            return 0

        fuel_val = 0

        if (action[0] == 'terminate') or (action[0] == 'reset'):
            return fuel_val

        new_state = self.taxi_agent.exec_action(state, action)
        for i, taxi in enumerate(state[0]):
            if taxi[3] == self.taxi_agent.initial["taxis"][taxi[0]]["fuel"]:
                continue
            if action[i][0] == "refuel":
                closest_passenger = self.closest_passenger(taxi[1], state[1], flag=False)
                if (closest_passenger[0] is not None) and \
                        (taxi[3] > (closest_passenger[1] +
                                    min(self.BFS_distance(closest_passenger[0][2], closest_passenger[0][3]),
                                        self.BFS_distance(closest_passenger[0][2], taxi[1])))):
                    continue
                fuel_val += 10
            elif (action[i][0] == "move") and (len(self.taxi_agent.locations_of_stations) > 0):
                min_dist_before = self.BFS_distance(state[0][i][1], self.taxi_agent.locations_of_stations[0])
                min_dist_after = self.BFS_distance(new_state[0][i][1], self.taxi_agent.locations_of_stations[0])
                for station in self.taxi_agent.locations_of_stations:
                    dist_before = self.BFS_distance(state[0][i][1], station)
                    dist_after = self.BFS_distance(new_state[0][i][1], station)
                    if dist_before < min_dist_before:
                        min_dist_before = dist_before
                    if dist_after < min_dist_after:
                        min_dist_after = dist_after

                if (min_dist_before - min_dist_after) != 0:
                    fuel_val += 1/(min_dist_before - min_dist_after)
        return fuel_val


    def man_distance(self, x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    def BFS_distance(self, p1, p2, flag=False):
        """
        :param p1: points 1 = (x, y)
        :param p2: point 2 = (x, y)
        :param flag: True if the solution is needed, False if only the distance is needed
        :return: the distance of the BFS path
        """

        if (p1 is None) or (p2 is None) or (len(p2) == 0):
            return -1
        # or (p2 in self.taxi_agent.)

        prob = mapProblem(
            {
                "location": p1,
                "map": self.taxi_agent.map,
            },
            p2
        )
        s = breadth_first_search(prob)

        if s is None:
            return -1

        solution = list(map(lambda n: n.action, s.path()))[1:]
        d = len(solution)
        if flag:
            return d, solution[-1]
        return d


class mapProblem:
    def __init__(self, initial, goal=None):
        self.initial = initial['location']
        self.goal = goal
        self.map = initial["map"]
        self.found_passenger = set()

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

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1


class Q_learning:
    def __init__(self, Q_: Q, taxiAgent: TaxiAgent):
        self.Q_ = Q_
        self.glie = GLIE(taxiAgent=taxiAgent)
        self.taxiAgent = taxiAgent

    def train(self, state, epochs=5000, timeout=300):
        s = copy.deepcopy(state)

        start = time.perf_counter()
        for i in range(epochs):
            if len(s) == 0:
                s = copy.deepcopy(state)
            a = self.glie.get_action(s, self.Q_)
            s_prime = self.taxiAgent.exec_action(s, a)
            self.Q_.train(s, s_prime, a)
            s = s_prime
            end = time.perf_counter()
            if end - start > timeout:
                return


class GLIE:
    def __init__(self, taxiAgent: TaxiAgent, T=15):
        self.T = T
        self.agent = taxiAgent

    def P(self, a, s, Q_: Q, pos_actions):
        maxExp = max([(Q_.calc(s, a_prime) / self.T) for a_prime in pos_actions])
        # print(Q_.calc(s, a), self.T)
        exps = sum([math.e ** ((Q_.calc(s, a_prime) / self.T) - maxExp) for a_prime in pos_actions])
        return (math.e ** ((Q_.calc(s, a) / self.T) - maxExp)) / exps

    def update_T(self, a=.999):
        # print(self.T)
        self.T = max(self.T * a, 0.25)

    def get_action(self, s, Q_: Q):
        pos_actions = self.agent.possible_actions(s)
        probs = {}
        for i, action in enumerate(pos_actions):
            temp = 0
            if i > 0:
                temp = probs[pos_actions[i - 1]]
            probs[action] = temp + self.P(action, s, Q_, pos_actions)

        self.update_T()

        p = random.random()

        for action, prob in probs.items():
            if p < prob:
                return action


from utils import FIFOQueue


class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


def breadth_first_search(problem):
    """[Figure 3.11]"""
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None







