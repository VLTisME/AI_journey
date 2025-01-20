# Author: Tuan Vo Lan
# Self-build simple CNN model for Move Prediction in Gomoku (follow 2016's paper)

import os
import sys
import hashlib
import numpy as np
import tensorflow as tf # type: ignore
from urllib.parse import unquote
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


BOARD_SIZE = 15
TEST_SIZE = 0.2
EPOCHS = 10


def main():

    print("LOADING DATA!...\n")
    states, actions = load_data()
    print("DONE data loading!\n")
    inputs = []

    for state in states:
        encoded_state = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32) 
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                token = state[i, j]
                if token == 0:
                    encoded_state[i, j] = [0, 0, 1]
                elif token == 1:
                    encoded_state[i, j] = [1, 0, 0]
                else:
                    encoded_state[i, j] = [0, 1, 0]
        inputs.append(encoded_state)
    
    actions = tf.keras.utils.to_categorical(actions, num_classes = BOARD_SIZE * BOARD_SIZE)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(inputs), np.array(actions), test_size = TEST_SIZE
    )

    model = get_model()
    model.fit(x_train, y_train, epochs = EPOCHS)
    model.evaluate(x_test, y_test, verbose = 2)
    model.save('gomoku_ai.h5')


def get_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, 3)),
            tf.keras.layers.Conv2D(32, (5, 5), activation = "relu"),
            *[tf.keras.layers.Conv2D(64, (3, 3)) for _ in range(4)],
            tf.keras.layers.Conv2D(128, (1, 1), activation = "relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(225, activation = "softmax")
    ])

    model.compile(
          optimizer = "adam",
          loss = "categorical_crossentropy",
          metrics = ["accuracy"]
    )
    
    return model



def hash_board(board):
    return hashlib.md5(board.tobytes()).hexdigest()  

def to_num(c):
    return int(ord(c) - ord('a'))

def load_data():
    file_path = os.path.join("renjunet_v10_20180803.xml")
    context = ET.iterparse(file_path, events = ("start", "end"))
    state = []
    action = []
    hash_values = set()
    # board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # flat_board = [0 for _ in range(BOARD_SIZE ** 2 + 2)]
    # Better using numpy array
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype = np.uint8)
    flat_board = np.zeros((BOARD_SIZE * BOARD_SIZE + 2), dtype = np.uint8)

    for event, elem in context:
        # I first train the bot with smaller data
        if len(state) > 5000:
            break
        if event == 'end':
           if elem.tag == 'move':
               text = elem.text.strip()
               moves = elem.text.strip().split()
               turn = 1
               ok = 0
               for move in moves:
                   if len(move) > 3: 
                      continue
                   row = to_num(move[0])
                   col = int(move[1:]) - 1
                   player = 2 if turn == 1 else 1 # 2 is X, 1 is O and 0 is an empty cell
                   flat_board[BOARD_SIZE * BOARD_SIZE] = row
                   flat_board[BOARD_SIZE * BOARD_SIZE + 1] = col
                   if ok == 1:
                      hash_value = hash_board(flat_board)
                      if hash_value not in hash_values:
                          hash_values.add(hash_value)
                          state.append(board.copy())
                          action.append(row * BOARD_SIZE + col)
                   board[row, col] = player
                   flat_board[row * BOARD_SIZE + col] = player
                   turn = -turn
                   ok = 1
               board.fill(0)
               flat_board.fill(0)
           elem.clear()
    return (state, action)


if __name__ == "__main__":
    main()