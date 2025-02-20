from tensorflow.keras.models import load_model # type: ignore
import numpy as np


board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

encoded_board = np.zeros((15, 15, 3), dtype=np.float32)  
decode = {}

cnt = 0
for i in range(15):
    for j in range(15):
        decode[cnt] = tuple((i, j))
        cnt = cnt + 1

for i in range(15):
    for j in range(15):
        if board[i][j] == 0:
            encoded_board[i, j] = [0, 0, 1]
        elif board[i][j] == 1:
            encoded_board[i, j] = [1, 0, 0]
        else:
            encoded_board[i, j] = [0, 1, 0]

encoded_board = np.expand_dims(encoded_board, axis=0)

model = load_model('gomoku_ai.h5')

prediction = model.predict(encoded_board)
print(len(prediction))
predicted_move = np.argmax(prediction)

print(f"Next move is: {decode[predicted_move]}")
