import math
import random
import time

# Okay so nếu như ta dùng @classmethod thì ta có thể gọi hàm đó mà không cần tạo instance của class (vd như là Nim.available_actions(piles)) còn nếu
# không dùng thì ta cần phải tạo instance của class trước rồi mới gọi hàm (vd như là nim = Nim() sau đó nim.available_actions(piles))
# Nó sẽ hữu dụng khi ta muốn tạo một hàm mà không cần phải tạo instance của class đó trước ví dụ như là một hàm dùng chung cho nhiều class
# ví dụ bên dưới thì class Nim() ta đang kiểm soát cả một trò chơi lớn nên chỉ cần gọi available_actions(piles) là trả về all actions rồi
# nch là thêm classmethod cho đỡ tốn công tạo instance của class đó mới gọi được.
# tóm tắt lại khi nào cần dùng @classmethod:
# - Khi muốn tạo một hàm dùng chung cho nhiều instance của class
# - Khi muốn tạo một hàm mà không cần phải tạo instance của class đó 
# Lưu ý là nếu có @classmethod thì phải dùng cls chứ không dùng self


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player): # phải là cls vì nó không cần instance của class đó để chạy được còn nếu 
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        return 0 if len(self.q) == 0 or (tuple(state), action) not in self.q else self.q[(tuple(state), action)]


    def update_q_value(self, state, action, old_q, reward, future_rewards):
        self.q[(tuple(state), action)] = old_q + self.alpha * ((reward + future_rewards) - old_q) # I added another theta for future rewards

    def best_future_reward(self, state):
        best_reward = -10000
        actions = Nim.available_actions(state)
        if len(actions) == 0:
            return 0
        for action in actions:
            best_reward = max(best_reward, self.get_q_value(tuple(state), action))
        return best_reward


    def choose_action(self, state, epsilon=True):
                
        actions = Nim.available_actions(state)
        if epsilon and random.random() <= epsilon:
            return random.choice(list(actions))
        else:
            best_action = None
            best_reward = None
            for action in actions:
                if best_reward == None or self.get_q_value(tuple(state), action) > best_reward:
                    best_reward = self.get_q_value(tuple(state), action)
                    best_action = action
            return best_action
         


def train(n):

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn") 
            pile, count = ai.choose_action(game.piles, epsilon=False) # either False or True if you want to use epsilon-greedy or just choosing the best action
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
