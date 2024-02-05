"""
This is a simple Tic-Tac-Toe game which uses MCTS and UCT in order to choose the next move.
The computer is the X player, and it plays first.
The user is the O player, and he reacts to the computer.

Written by Dvir Twito.
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple
from abc import ABC, abstractmethod

X_PLAYER = 1
O_PLAYER = -1
EMPTY_TILE = 0

X_MARK = 'X'
O_MARK = 'O'
EMPTY_MARK = ' '


class Board:
    def __init__(self, initial=None):
        super().__init__()
        if initial is None:
            self.tiles = [[EMPTY_TILE, EMPTY_TILE, EMPTY_TILE] for _ in range(3)]
        else:
            self.tiles = initial

    def __getitem__(self, item) -> int:
        row, col = item
        return self.tiles[row][col]

    def build_next_board(self, move, player) -> Board:
        row, col = move
        new_board = [row[:] for row in self.tiles]
        new_board[row][col] = player
        return Board(new_board)

    def get_result(self) -> int:
        # Check rows
        for row in range(3):
            if (self.tiles[row][0] == self.tiles[row][1] == self.tiles[row][2]) and self.tiles[row][0] != EMPTY_TILE:
                return self.tiles[row][0]

        # Check columns
        for col in range(3):
            if (self.tiles[0][col] == self.tiles[1][col] == self.tiles[2][col]) and self.tiles[0][col] != EMPTY_TILE:
                return self.tiles[0][col]

        # Check main diagonal
        if (self.tiles[0][0] == self.tiles[1][1] == self.tiles[2][2]) and self.tiles[0][0] != EMPTY_TILE:
            return self.tiles[0][0]

        # Check secondary diagonal
        if (self.tiles[0][2] == self.tiles[1][1] == self.tiles[2][0]) and self.tiles[0][2] != EMPTY_TILE:
            return self.tiles[0][2]

        return 0  # Tie

    def display(self):
        str_board = [[X_MARK if cell == X_PLAYER else O_MARK if cell == O_PLAYER else EMPTY_MARK for cell in row] for row in self.tiles]

        print('', ' | '.join(str_board[0]))
        print("---|---|---")
        print('', ' | '.join(str_board[1]))
        print("---|---|---")
        print('', ' | '.join(str_board[2]))


class State:
    def __init__(self, board=None, player=1):
        if board is None:
            self.board = Board()
        else:
            self.board = board
        self.player = player

    def make_move(self, move) -> State:
        new_board = self.board.build_next_board(move, self.player)
        next_player = -self.player
        return State(new_board, next_player)

    def get_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    moves.append((row, col))
        return moves

    def get_result(self) -> int:
        return self.board.get_result()

    def is_terminal(self):
        return self.get_result() != 0 or len(self.get_moves()) == 0


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []

    def add_child(self, child_state, move):
        child = Node(child_state, self, move)
        self.children.append(child)
        return child

    def select_child(self) -> Node:
        return max(self.children, key=lambda child: child.uct_value())

    def expand(self, moves):
        for move in moves:
            self.add_child(self.state.make_move(move), move)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


class Agent(ABC):
    @abstractmethod
    def get_move(self, state) -> Tuple[int, int]:
        raise NotImplementedError()


class UctAgent(Agent):
    def select(self, node):
        current_node = node
        while len(current_node.children) != 0:
            current_node = current_node.select_child()
        return current_node

    def simulate(self, node, player) -> int:
        """
        Preforms a random simulation
        """
        if node.state.is_terminal():
            return node.state.get_result()
        moves = node.state.get_moves()
        child_state = node.state.make_move(random.choice(moves))
        return self.simulate(Node(child_state, node), -player)

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def mcts(self, state, iterations=1000) -> Node:
        root = Node(state)
        for iteration in range(iterations):
            node = self.select(root)
            if node.state.is_terminal():
                self.backpropagate(node, node.state.get_result())
            else:
                node.expand(node.state.get_moves())
                result = self.simulate(node, node.state.player)
                self.backpropagate(node, result)
        return max(root.children, key=lambda child: child.wins / child.visits)  # No exploration

    def get_move(self, state) -> Tuple[int, int]:
        return self.mcts(state).move


class UserAgent(Agent):
    def get_move(self, state) -> Tuple[int, int]:
        row_str, col_str = input("\nEnter move (row col): ").split()
        row = int(row_str)
        col = int(col_str)
        return row, col


class TicTacToeGame:
    def __init__(self):
        self.mcts_agent = UctAgent()
        self.user_agent = UserAgent()

    def start(self):
        state = State()
        first_turn = True
        while not state.is_terminal():
            if not first_turn:
                print()
            else:
                first_turn = False
            print("Current board:")
            state.board.display()
            if state.player == X_PLAYER:
                move = self.mcts_agent.get_move(state)
                print(f"\nThe computer chose {move}")
            else:
                move = self.user_agent.get_move(state)
            state = state.make_move(move)
        print("\nFinal board:")
        state.board.display()
        result = state.get_result()
        print()
        if result > 0:
            print("You lost :(")
        elif result < 0:
            print("You won!")
        else:
            print("It's a tie...")


def main():
    game = TicTacToeGame()
    game.start()


if __name__ == "__main__":
    main()
