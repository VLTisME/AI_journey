import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    count_x = 0
    count_o = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                count_x += 1
            elif board[i][j] == O:
                count_o += 1
    if count_x == count_o:
        return X
    return O

def actions(board):
    possible_move = set()
    for i in range(3):
        for j in range (3):
            if board[i][j] == EMPTY:
                possible_move.add((i, j));
    return possible_move

def result(board, action):
    new_board = copy.deepcopy(board)
    if action[0] < 0 or action[1] < 0 or action[0] > 2 or action[1] > 2 or board[action[0]][action[1]] != EMPTY:
        raise ValueError("An error occurred with the board or move")
    
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    if board[0][0] == board[1][0] == board[2][0] or board[0][0] == board[0][1] == board[0][2] or board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] != EMPTY:
            return board[0][0]
    if board[1][1] == board[1][0] == board[1][2] or board[1][1] == board[0][2] == board[2][0] or board[1][1] == board[0][1] == board[2][1]:
        if board[1][1] != EMPTY:
            return board[1][1]
    if board[2][2] == board[2][1] == board[2][0] or board[2][2] == board[1][2] == board[0][2]:
        if board[2][2] != EMPTY:
            return board[2][2]
    return None


def terminal(board):
    if winner(board) != None:
        return True
    if len(actions(board)) == 0:
        return True
    return False

def utility(board):
    the_winner = winner(board)
    if the_winner == X:
        return 1
    elif the_winner == O:
        return -1
    return 0


def maximize(board):
    if terminal(board):
        return utility(board)
    res = -1
    possible_move = actions(board)
    for i, j in possible_move:
        res = max(res, minimize(result(board, (i, j))))
    return res

def minimize(board):
    if terminal(board):
        return utility(board)
    res = 1
    possible_move = actions(board)
    for i, j in possible_move:
        res = min(res, maximize(result(board, (i, j))))
    return res

def minimax(board):
    if terminal(board) == True:
        return None
    possible_move = actions(board)
    best_action = (0, 0)
    if player(board) == X:
       res = -1
       for i, j in possible_move:
           uti = minimize(result(board, (i, j)))
           if uti >= res:
               res = uti
               best_action = (i, j)
    else:
        res = 1
        for i, j in possible_move:
            uti = maximize(result(board, (i, j)))
            if uti <= res:
                res = uti
                best_action = (i, j)
              
    return best_action
               
       
