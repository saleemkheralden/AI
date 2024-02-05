import math
import random
import time
from copy import deepcopy
import numpy as np
import itertools
from Simulator import Simulator

IDS = ["212699581"]

class UCTNode:
    def __init__(self, state=None, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []

    def score(self):
        return self.wins / self.visits if self.visits > 0 else 0

    def select(self, t=1):
        return max(self.children, key=lambda child: child.uct_value(t=t))

    def uct_value(self, t=1):
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + math.sqrt(2 * math.log(t) / self.visits)

    def expand(self, actions, result_func):
        # self.state = deepcopy(self.state)
        for e in actions:
            self.children.append(UCTNode(state=result_func(e), parent=self, move=e))

    def update(self, result):
        self.wins += result
        self.visits += 1

# class UCTTree:
#     def __init__(self, root=None):
#         self.root = root
#
#         if root is not None:
#             self.nodes = [root]
#             self.leaves = [root]
#
#     def expand(self, parent_node: UCTNode, actions, result_func):
#         parent_node.expand(actions, result_func)
#         if parent_node in self.leaves:
#             self.leaves.remove(parent_node)
#         self.leaves += parent_node.children
#         self.nodes += parent_node.children

class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.taxis = []
        self.opp_taxis = []
        self.t = 1
        self.simulator = Simulator(initial_state)
        self.player_number = player_number
        self.map = initial_state['map']
        # self.uct_tree = UCTTree(initial_state)

        for taxi_name, taxi in initial_state['taxis'].items():
            if taxi['player'] == player_number:
                self.taxis.append(taxi_name)
            else:
                self.opp_taxis.append(taxi_name)

    def print_game(self, state):
        board = []


        for row in self.map:
            row_print = []
            for col in row:
                if col == 'I':
                    row_print.append('@@|')
                else:
                    row_print.append('  |')
            board.append(row_print)
            board.append(['---'] * len(row))

        for taxi_name, taxi in state['taxis'].items():
            board[2 * taxi['location'][0]][taxi['location'][1]] = f'T{taxi_name.split(" ")[1]}|'

        for passenger_name, passenger in state['passengers'].items():
            if type(passenger['location']) == str:
                continue
            board[2 * passenger['location'][0]][passenger['location'][1]] = f'P{passenger_name[0]}|'

        print('   ', end='')
        print(*[f'{i:<3}' for i in range(len(board[0]))], sep='')
        for row_idx, row in enumerate(board):
            if row_idx % 2 == 0:
                print(f'{int(row_idx / 2):<3}', end='')
            else:
                print('   ', end='')
            print(*row, sep='')

    def selection(self, UCT_tree):
        node = UCT_tree
        while len(node.children) != 0:
            node = node.select(t=self.t)
        return node

    def actions(self, state, player_num, opp=False):
        atomic_actions = []
        locations = set()
        simulator = Simulator(state)

        taxis = self.opp_taxis if opp else self.taxis
        for taxi in taxis:
            atomic_actions.append([])
            locations.add(state["taxis"][taxi]["location"])

            atomic_actions[-1] += [('move', taxi, location)
                                   for location in self.simulator.neighbors(state['taxis'][taxi]['location'])]
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger_name, passenger in state["passengers"].items():
                    if passenger["location"] == state["taxis"][taxi]["location"]:
                        atomic_actions[-1].append(("pick up", taxi, passenger_name))

            for passenger_name, passenger in state["passengers"].items():
                if (passenger["destination"] == state["taxis"][taxi]["location"]
                        and passenger["location"] == taxi):
                    atomic_actions[-1].append(("drop off", taxi, passenger_name))
            atomic_actions[-1].append(("wait", taxi))

        actions = []
        for comb in itertools.product(*atomic_actions):
            if simulator.check_if_action_legal(comb, player_num):
                actions.append(comb)
        return actions

    def expansion(self, UCT_tree, parent_node):


        state = UCT_tree.state

        def result(action):
            simulator = Simulator(state)
            simulator.apply_action(action, self.player_number)
            return deepcopy(simulator.state)

        UCT_tree.expand(self.actions(UCT_tree.state, self.player_number), result)

    def simulation(self, node: UCTNode, timeout: float = 5):
        start = time.perf_counter()
        simulator = Simulator(node.state)
        player = self.player_number

        # print('starting simulation')
        # self.print_game(node.state)


        while time.perf_counter() - start < timeout:
        # for i in range(int(timeout * 10)):
            moves = self.actions(node.state, player, opp=(player != self.player_number))
            if len(moves) == 1:
                return
            move = random.choice(moves)
            # print(f"player: {player} made the move {move}")
            simulator.apply_action(move, player)
            # self.print_game(simulator.get_state())

            player = 1 if player == 2 else 2
            node = UCTNode(state=deepcopy(simulator.get_state()), parent=node)

        return simulator.get_score()

    def backpropagation(self, node, simulation_result):
        simulation_result = simulation_result[f'player {self.player_number}']
        while node is not None:
            # node.update(1 if simulation_result > 0 else 0)
            node.update(simulation_result)
            node = node.parent


    def mcts(self, state, timeout=4.9):
        root = UCTNode(state)
        # uct_tree = UCTTree(root)
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
        # for i in range(int(timeout * 10)):
            node = self.selection(root)
            self.expansion(node, node)
            result = self.simulation(node, timeout=timeout-(time.perf_counter() - start) - 0.2)
            self.backpropagation(node, result)
        self.t += 1
        return max(root.children, key=lambda child: child.score())



    def act(self, state):
        return self.mcts(state).move


class Agent(UCTAgent):
    def __init__(self, initial_state, player_number):
        super().__init__(initial_state, player_number)



