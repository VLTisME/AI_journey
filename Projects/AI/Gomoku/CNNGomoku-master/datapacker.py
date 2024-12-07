import torch.utils.data as data
import torch
import numpy as np
from SGFFileProcess import *
sgf = SGFflie()

def dataloader(path, MINIBATCH_SIZE):

    # angle = random.choice([ -90, 0, 90, 180])
    # img = TF.rotate(img, angle)

    board, nextmove = sgf.createTraindata(path)

    # reshape data from here.
    x = np.array(board).reshape(len(board), 1, 15, 15)
    x = torch.from_numpy(x)
    for i in range(len(nextmove)):
        nextmove[i] = np.argmax(nextmove[i])
    y = torch.from_numpy(np.array(nextmove))


    #pack to tensor dataset
    torch_dataset = data.TensorDataset(x, y)

    # put the dataset into DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )


    print("\ndata loaded, board nums: ", len(board))
    return loader

def augumentation(x, y):
    #data augumentation, only 8 possiblities
    #1. flip along y
    x1 = x.flip(3)
    y1 = y.flip(3)
    # 2. rotate 90 and flip along x or not.
    x2 = x.transpose(2, 3).flip(3)
    y2 = y.transpose(2, 3).flip(3)

    x3 = x.transpose(2, 3).flip(3).flip(2)
    y3 = y.transpose(2, 3).flip(3).flip(2)

    # 3. rotate 180 and flip along y or not.
    x4 = x.flip(2).flip(3)
    y4 = y.flip(2).flip(3)

    x5 = x.flip(2)
    y5 = y.flip(2)

    # 4. rotate -90 and flip along x or not.

    x6 = x.transpose(2, 3).flip(2)
    y6 = y.transpose(2, 3).flip(2)

    x7 = x.transpose(2, 3)
    y7 = y.transpose(2, 3)
    x = torch.cat((x,x1,x2,x3,x4,x5,x6,x7),0)
    y = torch.cat((y,y1,y2,y3,y4,y5,y6,y7),0)
    return x, y