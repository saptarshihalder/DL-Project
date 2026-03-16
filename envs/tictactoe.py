import numpy as np


class TicTacToe:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        state[:, :, 0] = (self.board == 1).astype(np.float32)
        state[:, :, 1] = (self.board == 2).astype(np.float32)
        state[:, :, 2] = (self.board == 0).astype(np.float32)
        return state

    def get_valid_actions(self):
        actions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 0:
                    actions.append(r * self.cols + c)
        return actions

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over")

        row = action // self.cols
        col = action % self.cols

        if self.board[row][col] != 0:
            raise ValueError(f"Position ({row}, {col}) is already taken")

        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            reward = 0.0

        info = {"winner": self.winner, "current_player": self.current_player}
        self.current_player = 3 - self.current_player

        return self.get_state(), reward, self.done, info

    def _check_win(self, player):
        b = self.board
        for r in range(3):
            if b[r][0] == b[r][1] == b[r][2] == player:
                return True
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] == player:
                return True
        if b[0][0] == b[1][1] == b[2][2] == player:
            return True
        if b[0][2] == b[1][1] == b[2][0] == player:
            return True
        return False

    def render(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        print("-" * 7)
        for r in range(3):
            print("|", " ".join(symbols[self.board[r][c]] for c in range(3)), "|")
        print("-" * 7)
