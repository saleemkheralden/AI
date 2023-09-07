import copy
import itertools
import math
import random
import numpy as np

import utils
import time

ids = ["111111111", "222222222"]


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

    def possible_destinations(self, state):
        location_combinations = []
        for passenger in self.initial["passengers"].keys():
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
                            prob *= ((1 - pass_prob) + (pass_prob / (
                                len(self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"]))))
                        else:
                            prob *= (1 - pass_prob)
                    else:
                        prob *= (pass_prob / (
                            len(self.initial["passengers"][self.index_to_passenger[i]]["possible_goals"])))
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
                if len(taxi[2]) < self.initial["taxis"][taxi[0]]["capacity"] and p[1] == "no taxi" and p[2] == taxi[
                    1] and p[2] != p[3]:
                    curr_states.append(("pick up", taxi[0], p[0]))

            for p in taxi[2]:
                if taxi[1] == state[1][self.passenger_to_index[p]][3]:
                    curr_states.append(("drop off", taxi[0], p))

            if taxi[1][0] > 0 and self.initial["map"][taxi[1][0] - 1][taxi[1][1]] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0] - 1, taxi[1][1])))

            if taxi[1][0] < len(self.initial["map"]) - 1 and self.initial["map"][taxi[1][0] + 1][taxi[1][1]] != "I" and \
                    taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0] + 1, taxi[1][1])))

            if taxi[1][1] > 0 and self.initial["map"][taxi[1][0]][taxi[1][1] - 1] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0], taxi[1][1] - 1)))

            if taxi[1][1] < len(self.initial["map"][0]) - 1 and self.initial["map"][taxi[1][0]][
                taxi[1][1] + 1] != "I" and taxi[3] > 0:
                curr_states.append(("move", taxi[0], (taxi[1][0], taxi[1][1] + 1)))

            actions_per_taxis.append(curr_states)

        combinations = []
        for comb in itertools.product(*actions_per_taxis):
            curr_locations = set()
            for i in range(len(state[0])):
                curr_locations.add(state[0][i][1])
            for taxi in comb:
                if taxi[0] == "move":
                    # print(curr_locations, comb, )
                    curr_locations.remove(state[0][self.taxi_to_index[taxi[1]]][1])
                    curr_locations.add(taxi[2])
            if len(curr_locations) == len(state[0]):
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
        # print(state)
        # print(self.val_func[state])
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
        print("starting __init__")
        super().__init__(initial, flag=False)
        self.initial = initial

        # print(self.initial_state)

        self.q_learning = Q_learning(self)
        # self.q_learning.train(self.initial)

        self.q_learning.train(self.initial_state)

        # self.t = thread()
        # self.t.update_func(self.q_learning.train)
        # self.t.update_params(self.initial_state)
        #
        # self.t.start()
        # self.t.join(timeout=299.9)
        print("finished __init__")
        self.locations_of_stations = []
        for i, row in enumerate(initial["map"]):
            for j, col in enumerate(row):
                if col == "G":
                    self.locations_of_stations.append((i, j))

    def act(self, state, flag=True):
        self.possible_actions(state)

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

    #
    # def act(self, state, flag=True):
    #     print("starting act: ")
    #     # start = time.perf_counter()
    #     # print(start)
    #     taxis_pass = {}
    #     for taxi in state["taxis"].keys():
    #         taxis_pass[taxi] = []
    #     for pas in state["passengers"].keys():
    #         if type(state["passengers"][pas]["location"]) == str:
    #             taxis_pass[state["passengers"][pas]["location"]].append(pas)
    #     taxis = [[i, state["taxis"][i]["location"], taxis_pass[i], state["taxis"][i]["fuel"]]
    #              for i in state["taxis"].keys()]
    #     passengers = [[i,
    #                    state["passengers"][i]["location"] if type(state["passengers"][i]["location"]) == str
    #                    else "no taxi",
    #                    state["passengers"][i]["location"],
    #                    state["passengers"][i]["destination"]] for i in
    #                   state["passengers"].keys()]
    #     state_arr = [taxis, passengers, state["turns to go"]]
    #     state_tuple = self.array_to_tuple(state_arr)
    #
    #     if state_tuple in self.val_func.keys():
    #         x = self.val_func[state_tuple][0]
    #         if x[0] == "reset" or x[0] == "terminate":
    #             return x[0]
    #         return x
    #     # else:
    #     #     self.t.update_params(state_arr)
    #     #     self.t.join(timeout=0.05)
    #     #     if state_tuple in self.val_func.keys():
    #     #         x = self.val_func[state_tuple][0]
    #     #         if x[0] == "reset" or x[0] == "terminate":
    #     #             return x[0]
    #     #         return x
    #     #     else:
    #     #         print('no luck')
    #     #         print('will start greedy')
    #
    #     actions = self.possible_actions(state_tuple)
    #
    #     points = {}
    #
    #     for action in actions:
    #         points[action] = self.calc_action(state_arr, action)
    #     # end = time.perf_counter()
    #     # print(end)
    #     # print(end - start)
    #     return max(points, key=points.get)


class Q:
    def __init__(self, taxi_agent: TaxiAgent, lr=.9, gamma=.5):
        self.features = [lambda *args: 1, self.taxi_locations, self.passenger_locations]
        # self.theta = [0] * len(self.features)
        self.theta = np.random.normal(0, 1, len(self.features))
        self.taxi_agent = taxi_agent

        self.fuel_thetas = [0] * len(taxi_agent.initial_state[0])
        self.lr = lr
        self.gamma = gamma

        # self.max_q_table = {}

    def calc(self, state, action):
        possible_destinations = self.taxi_agent.possible_destinations(state)
        return sum([theta_i * f(state, action, possible_destinations) for theta_i, f in zip(self.theta, self.features)])

    def maxQ(self, state):

        # if state in self.max_q_table.keys():
        #     return self.max_q_table[state]

        actions = self.taxi_agent.possible_actions(state)
        maxq = self.calc(state, actions[0])

        for action in actions[1:]:
            tempCalc = self.calc(state, action)
            if tempCalc > maxq:
                maxq = tempCalc

        # self.max_q_table[state] = maxq

        return maxq

    def train(self, s, s_prime, a):
        for i, theta_i in enumerate(self.theta):
            self.theta[i] = theta_i + self.lr * \
                            (0 + self.gamma * self.maxQ(s_prime) - self.calc(s, a)) * \
                            self.features[i](s, a)

    def closest_passenger(self, corr, passengers, index=2):
        min_dist = float('inf')
        min_passenger = None
        for passenger in passengers:
            if type(passenger[index]) is str:
                continue

            dist = self.man_distance(corr, passenger[index])
            if dist < min_dist:
                min_dist = dist
                min_passenger = passenger
        return min_passenger, min_dist

    def calc_action(self, state, action, possible_destinations):
        points = 0
        for taxi in state[0]:

            # get the corresponding action to the taxi
            taxi_action = None
            for atom_action in action:
                if atom_action[0] == "reset":
                    points -= 50
                    break
                elif atom_action[0] == "terminate":
                    break
                if atom_action[1] == taxi[0]:
                    taxi_action = atom_action
                    break

            if taxi_action is None:
                continue

            if taxi_action[0] == "move":

                closest_passenger, passenger_dist = self.closest_passenger(taxi[1], state[1])
                curr_state = copy.deepcopy(state)

                for dest in possible_destinations:
                    prob = dest[0]
                    for i, passenger_dest in enumerate(dest[1]):
                        curr_state[1][i][3] = passenger_dest
                    if len(taxi[2]) > 0:
                        closest_dest, dest_dist = self.closest_passenger(taxi[1], curr_state[1], 3)
                        if dest_dist < passenger_dist:
                            if self.man_distance(taxi_action[2], closest_dest[3]) < \
                                    self.man_distance(taxi[1], closest_dest[3]):
                                points += prob
                                continue
                    if closest_passenger is not None:
                        if self.man_distance(taxi_action[2], closest_passenger[2]) < \
                                self.man_distance(taxi[1], closest_passenger[2]):
                            points += prob
            elif taxi_action[0] == "drop off":
                points += 100
            elif taxi_action[0] == "pick up":
                points += 5
            elif taxi_action[0] == "refuel":
                points += -10
            else:
                points += 0

        return points

    def calc_utility(self, state):
        locations_of_taxis = []
        utility = 0
        for taxi in state[0]:
            locations_of_taxis.append(taxi[1])
        for passenger in state[1]:
            closest_taxi = self.man_distance(passenger[2], locations_of_taxis[0])
            for taxi in locations_of_taxis:
                curr_dist = self.man_distance(taxi, passenger[2])
                if curr_dist < closest_taxi:
                    closest_taxi = curr_dist
            utility += closest_taxi
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
        fuel_val = 0
        new_state = self.taxi_agent.exec_action(state, action)
        for i, taxi in enumerate(state[0]):
            if action[i][0] == "refuel":
                fuel_val += 10
            elif action[i][0] == "move":
                min_dist_before = self.man_distance(state[0][i][0], self.taxi_agent.locations_of_stations[0])
                min_dist_after = self.man_distance(new_state[0][i][0], self.taxi_agent.locations_of_stations[0])
                for station in self.taxi_agent.locations_of_stations:
                    dist_before = self.man_distance(state[0][i][0], station)
                    dist_after = self.man_distance(new_state[0][i][0], station)
                    if dist_before < min_dist_before:
                        min_dist_before = dist_before
                    if dist_after < min_dist_after:
                        min_dist_after = dist_after
                fuel_val += 1/(min_dist_before - min_dist_after)
        return fuel_val

    def man_distance(self, x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])


class Q_learning:
    def __init__(self, taxiAgent: TaxiAgent):
        self.Q_ = Q(taxi_agent=taxiAgent)
        self.glie = GLIE(taxiAgent=taxiAgent)
        self.taxiAgent = taxiAgent

    def train(self, state, epochs=1000):
        s = copy.deepcopy(state)

        for i in range(epochs):
            a = self.glie.get_action(s, self.Q_)
            self.Q_.train(s, self.taxiAgent.exec_action(s, a), a)


class GLIE:
    def __init__(self, taxiAgent: TaxiAgent, T=5):
        self.T = T
        self.agent = taxiAgent

    def P(self, a, s, Q_: Q, pos_actions):
        exps = sum([math.e ** (Q_.calc(s, a_prime) / self.T) for a_prime in pos_actions])
        return (math.e ** (Q_.calc(s, a) / self.T)) / exps

    def update_T(self, a=.9):
        self.T = self.T * a

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
