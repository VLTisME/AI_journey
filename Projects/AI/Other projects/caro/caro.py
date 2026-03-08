import copy
import random
import time

class Caro():
    def __init__(self, size=6, straight=4):
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.board_size = size
        self.player = 0
        self.straight = straight
        self.winner = None

    @classmethod
    def available_actions(cls, board):
        actions = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == None:
                    actions.add((i, j))
        return actions
    
    @classmethod
    def other_player(cls, current_player):
        return 1 if current_player == 0 else 0
        
    def switch_player(self):
        self.player = Caro.other_player(self.player)
    
    def valid_cell(self, action):
        x, y = action
        if x >= self.board_size or x < 0 or y >= self.board_size or y < 0:
            return 0
        return 1
    
    def is_terminated(self, action):
        x, y = action
        p = self.player
        
        def count_line(dx, dy):
            cnt = 1
            i = 1
            while self.valid_cell((x + i * dx, y + i * dy)) and self.board[x + i * dx][y + i * dy] == p:
                cnt += 1
                i += 1

            i = 1
            while self.valid_cell((x - i * dx, y - i * dy)) and self.board[x - i * dx][y - i * dy] == p:
                cnt += 1
                i += 1
            return cnt
        if any([
            count_line(1, 0) >= self.straight,
            count_line(0, 1) >= self.straight,
            count_line(1, 1) >= self.straight,
            count_line(1, -1) >= self.straight
        ]):
            return 1  # winner

        if all(self.board[i][j] is not None for i in range(self.board_size) for j in range(self.board_size)):
            return 2  # tie

        return 0  # continue
            
    
    def move(self, action):
        x, y = action
        if self.winner is not None:
            raise Exception("Game has ended!")
        elif not self.valid_cell(action) or self.board[x][y] is not None:
            raise Exception("Invalid move!")
        
        self.board[x][y] = self.player
        status = self.is_terminated(action)
        
        if status == 1:
            self.winner = self.player
        elif status == 2:
            self.winner = -1

# Q-Learning
class CaroAI():
    def __init__(self, alpha=0.5, gamma = 0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = dict()
        
    def _state_key(self, state):
        return tuple(tuple(row) for row in state)

    def get_future_reward(self, state):
        actions = Caro.available_actions(state)
        return max([self.get_q_value(state, action) for action in actions], default=0)
    # Nim ends when there is no available actions, which also means someone wins
    # unlike Nim, Caro can end when there is still available actions (someone wins) -> we need to add terminated to ensure when the game is terminated, the future reward is 0, not the max of available actions which is not exist at that time
    def update(self, state, action, new_state, reward, terminated=False):
        old_q = self.get_q_value(state, action)
        next_q = self.get_future_reward(new_state) if not terminated else 0
        self.update_q_value(state, action, old_q, reward, next_q)
    
    def update_q_value(self, state, action, old_q, reward, next_q):
        theta = reward + self.gamma * next_q - old_q
        self.q[(self._state_key(state), action)] = old_q + self.alpha * theta
        
    def get_q_value(self, state, action):
        return 0 if len(self.q) == 0 or (self._state_key(state), action) not in self.q else self.q[(self._state_key(state), action)]

    def choose_action(self, state, epsilon=True):
        actions = list(Caro.available_actions(state))
        if epsilon and random.random() <= self.epsilon and len(actions) > 0:
            return random.choice(actions)
        else:
            best_reward = float("-inf")
            best_action = None
            for action in actions:
                reward = self.get_q_value(state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
            return best_action
        
def train(n):
    player = CaroAI()
    
    for i in range(n):
        if i % 5000 == 0:
            print(f"Playing training game {i + 1}")
        game = Caro()
        
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }
        
        while True:
            state = copy.deepcopy(game.board)
            action = player.choose_action(state, epsilon=True)
            
            last[game.player]["state"] = state
            last[game.player]["action"] = action
            
            if action is None:
                break
            
            game.move(action)
            game.switch_player()
            new_state = copy.deepcopy(game.board)
            
            if game.winner is not None and game.winner != -1:
                player.update(state, action, new_state, 1, terminated=True)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    -1,
                    terminated=True
                )
                break
            else: # TODO: why differ here?
                if game.winner == -1:
                    player.update(state, action, new_state, 0, terminated=True)
                else:
                    player.update(state, action, new_state, 0, terminated=False)
    
    print("Done training")
    
    return player

def play(ai, human_player=None):
    if human_player is None:
        human_player = random.randint(0, 1)
    
    game = Caro()
    
    while True:
        
        # visulize board
        print("\nCurrent board:")
        for i in range(game.board_size):
            for j in range(game.board_size):
                cell = game.board[i][j]
                if cell is None:
                    print(".", end=" ")
                elif cell == 0:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()
        print()
        
        available_actions = Caro.available_actions(game.board)
        time.sleep(1)
        if game.player == human_player:
            print("Your turn!")
            while True:
                try:
                    x = int(input("Enter row: "))
                    y = int(input("Enter column: "))
                    action = (x, y)
                    if action in available_actions:
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter valid integers for row and column.")
        else:
            print("AI's turn!")
            action = ai.choose_action(copy.deepcopy(game.board), epsilon=False)
            
        game.move(action)
        game.switch_player()
        
        if game.winner is not None:
            if game.winner == human_player:
                print("Congratulations! You win!")
            elif game.winner == -1:
                print("It's a tie!")
            else:
                print("AI wins! Better luck next time.")
            break

if __name__ == "__main__":
    ai = train(2000000)
    i = 0
    while True:
        print(f"Playing game {i + 1}")
        play(ai)
        i += 1
        print("Play again? (y/n)")
        if input().lower() != "y":
            break