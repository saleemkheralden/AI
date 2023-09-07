import copy
import itertools
import math
import random
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

    def act(self, state):
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


class TaxiAgent:
    def __init__(self, initial):
        #hyper parameters:
        self.T = 10
        self.alpha = 0.01
        self.gamma = 0.8
        self.features = [lambda *args: 1, self.f1, self.f2, self.f3, self.f4, self.f5]
        self.num_of_features = len(self.features)
        self.theta = [random.random() for _ in range(self.num_of_features)]
        self.initial = initial
        self.taxis = [[i, self.initial["taxis"][i]["location"], [], self.initial["taxis"][i]["fuel"]] for i in
                      self.initial["taxis"].keys()]
        self.passengers = [
            [i, "no taxi", self.initial["passengers"][i]["location"], self.initial["passengers"][i]["destination"]] for
            i in self.initial["passengers"].keys()]
        self.initial_state = [self.taxis, self.passengers, self.initial["turns to go"]]
        self.taxi_to_index = {taxi: idx for idx, taxi in enumerate(self.initial["taxis"].keys())}
        self.index_to_passenger = {idx: passenger for idx, passenger in enumerate(self.initial["passengers"].keys())}
        self.passenger_to_index = {passenger: idx for idx, passenger in enumerate(self.initial["passengers"].keys())}
        self.locations_of_stations = []
        for i, row in enumerate(initial["map"]):
            for j, col in enumerate(row):
                if col == "G":
                    self.locations_of_stations.append((i, j))

        self.train(self.initial_state)

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

    def possible_actions(self, state, train = True):
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
            curr_locations = []
            for i in range(len(state[0])):
                curr_locations.append(state[0][i][1])
            for taxi in comb:
                if taxi[0] == "move":
                    curr_locations.remove(state[0][self.taxi_to_index[taxi[1]]][1])
                    curr_locations.append(taxi[2])
            if len(set(curr_locations)) == len(curr_locations):
                combinations.append(comb)
        combinations.append(("reset",))
        # if not train:
        #     combinations.append(("terminate",))

        return combinations

    def calc_utility(self, state):
        locations_of_taxis = []
        utility = 0
        for taxi in state[0]:
            locations_of_taxis.append(taxi[1])
        for passenger in state[1]:
            if passenger[1] == "no taxi":
                closest_taxi = self.man_distance(passenger[2], locations_of_taxis[0])
                for taxi in locations_of_taxis:
                    curr_dist = self.man_distance(taxi, passenger[2])
                    if curr_dist < closest_taxi:
                        closest_taxi = curr_dist
                utility += closest_taxi
        return utility

    def man_distance(self, x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    def f1(self, state, action):
        new_state = self.exec_action(state, action)
        val = self.calc_utility(state) - self.calc_utility(new_state)
        return val

    def f2(self, state, action):
        if len(self.locations_of_stations) == 0:
            return 0
        fuel_val = 0

        if (action[0] == 'terminate') or (action[0] == 'reset'):
            return fuel_val

        new_state = self.exec_action(state, action)
        for i, taxi in enumerate(state[0]):
            if taxi[3] > self.initial["taxis"][taxi[0]]["fuel"] / 2:
                continue
            elif action[i][0] == "refuel":
                fuel_val += 10
            elif action[i][0] == "move":
                min_dist_before = self.man_distance(state[0][i][1], self.locations_of_stations[0])
                min_dist_after = self.man_distance(new_state[0][i][1], self.locations_of_stations[0])
                for station in self.locations_of_stations:
                    dist_before = self.man_distance(state[0][i][1], station)
                    dist_after = self.man_distance(new_state[0][i][1], station)
                    if dist_before < min_dist_before:
                        min_dist_before = dist_before
                    if dist_after < min_dist_after:
                        min_dist_after = dist_after
                fuel_val += min_dist_after - min_dist_before
        return fuel_val

    def f3(self, state, action):
        val = 0
        for sub_action in action:
            if sub_action[0] == "pick up":
                val += 2
            if sub_action[0] == "drop off":
                val += 20
        # val += self.f4(state,action) + self.f5(state,action)

        return val
        # return self.cal_reward(action)

    def f4(self, state, action):
        next_state = self.exec_action(state, action)
        # total_curr = 0
        # total_next = 0
        total = 0
        for i in range(len(state[0])):
            if len(state[0][i][2]) > 0:
                continue
            closest_passenger_curr = len(self.initial["map"]) * 2
            closest_passenger_next = len(self.initial["map"]) * 2
            for passenger in state[1]:
                if passenger[1] == "no taxi" and passenger[2] != passenger[3]:
                    curr_dist = self.man_distance(state[0][i][1], passenger[2])
                    next_dist = self.man_distance(next_state[0][i][1], passenger[2])
                    if curr_dist < closest_passenger_curr:
                        closest_passenger_curr = curr_dist
                    if next_dist < closest_passenger_next:
                        closest_passenger_next = next_dist
            if closest_passenger_next != closest_passenger_curr:
                total += (closest_passenger_curr - closest_passenger_next)
            # total_curr += closest_passenger_curr
            # total_next += closest_passenger_next
        return total

    def f5(self, state, action):
        if action[0] == "reset" or action[0] == "terminate":
            return 0
        next_state = self.exec_action(state,action)
        next_val = 0
        curr_val = 0
        curr_possible_destinations = self.possible_destinations(state)
        next_possible_destinations = self.possible_destinations(next_state)
        for i, taxi in enumerate(state[0]):
            if action[i][0] == "move" and len(taxi[2]) > 0:
                # print(1)
                closest_dest = len(self.initial["map"]) * 2
                for passenger in taxi[2]:
                    curr_dist = 0
                    for prob, dest in curr_possible_destinations:
                        curr_dist += prob * self.man_distance(taxi[1], dest[self.passenger_to_index[passenger]])
                    if curr_dist < closest_dest:
                        closest_dest = curr_dist
                curr_val += closest_dest
        for i, taxi in enumerate(next_state[0]):
            if action[i][0] == "move" and len(taxi[2]) > 0:
                closest_dest = len(self.initial["map"]) * 2
                for passenger in taxi[2]:
                    curr_dist = 0
                    for prob, dest in next_possible_destinations:
                        curr_dist += prob * self.man_distance(taxi[1], dest[self.passenger_to_index[passenger]])
                    if curr_dist < closest_dest:
                        closest_dest = curr_dist
                next_val += closest_dest
        # print(curr_val - next_val)
        return curr_val - next_val

    def cal_reward(self, action):
        if action[0] == "reset":
            return -50
        if action[0] == "terminate":
            return 0

        sum_reward = 0
        for sub_action in action:
            if sub_action[0] == "drop off":
                sum_reward += 100
            elif sub_action[0] == "refuel":
                sum_reward -= 10
        return sum_reward

    def Q_func(self, state, action, default=True):
        if action[0] == "terminate":
            return 0
        Q_val = []
        for i, feature in enumerate(self.features):
            Q_val.append(feature(state, action) * self.theta[i])
        if default:
            return sum(Q_val)
        return Q_val

    def maxQ(self, state, default):
        # if len(state) == 0:
        #     return 0
        actions = self.possible_actions(state)
        maxq = self.Q_func(state, actions[0], False)
        max_act = actions[0]
        for action in actions:
            tempCalc = self.Q_func(state, action, False)
            if sum(tempCalc) > sum(maxq):
                maxq = tempCalc
                max_act = action

        return maxq[default] + self.cal_reward(max_act)

    def train(self, state, duration_time=20):
        time1 = time.time()
        time2 = time.time()
        s = copy.deepcopy(state)

        while time2 - time1 < duration_time:
            a = self.GLIE_policy(state)
            next_s = self.exec_action(s, a)
            for i in range(self.num_of_features):
                fi = self.features[i](s, a)
                # self.theta[i] = self.theta[i] - self.alpha * ()
                self.theta[i] = self.theta[i] - self.alpha * (
                            -self.cal_reward(a) + self.gamma * self.maxQ(next_s, i) - fi) * fi
                if self.theta[i] > 10:
                    self.theta[i] = 10
                # self.theta[i] = self.theta[i] + self.alpha * (self.cal_reward(a) + self.gamma * self.maxQ(next_s,i) - fi) * fi
            # sum_theta = sum([abs(e) for e in self.theta])
            # for i, theta_i in enumerate(self.theta):
            #     self.theta[i] = theta_i / sum_theta
            #     if self.theta[i] < 0.1 and self.theta[i] > -0.1:
            #         self.theta[i] = self.theta[i]
            time2 = time.time()

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
            if task[0] == "drop off":
                curr_state[0][self.taxi_to_index[task[1]]][2].remove(task[2])
                curr_state[1][self.passenger_to_index[task[2]]][1] = "no taxi"
                curr_state[1][self.passenger_to_index[task[2]]][2] = curr_state[0][self.taxi_to_index[task[1]]][1]
            if task[0] == "move":
                curr_state[0][self.taxi_to_index[task[1]]][1] = task[2]
                curr_state[0][self.taxi_to_index[task[1]]][3] -= 1
        return curr_state

    def act(self, state):
        # self.theta = [0,0.1,0.1,0.95,2,10]
        # print(self.theta)
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
                      state["passengers"].keys()]
        state_arr = [taxis, passengers, state["turns to go"]]

        pos_actions = self.possible_actions(state_arr, train = False)

        maxq = self.Q_func(state_arr, pos_actions[0])
        maxAction = pos_actions[0]

        for action in pos_actions[1:]:
            tempCalc = self.Q_func(state_arr, action)
            if tempCalc > maxq:
                maxq = tempCalc
                maxAction = action
        # print(self.theta)
        # print("\n",maxAction, state_arr)
        if maxAction[0] == "reset" or maxAction[0] == "terminate":
            return maxAction[0]

        return maxAction



    def GLIE_policy(self, state):
        possible_actions = self.possible_actions(state)
        # print(self.theta)
        arr = [math.e ** (self.Q_func(state, a) / self.T) for a in possible_actions]
        exp = sum(arr)
        pr = {}
        curr_prob = 0
        for a in possible_actions:
            curr_prob += (math.e ** (self.Q_func(state, a) / self.T)) / exp
            pr[a] = curr_prob
        if self.T > 0.3:
            self.T = self.T * 0.95

        r = random.random()
        for a, prob in pr.items():
            if prob >= r:
                return a