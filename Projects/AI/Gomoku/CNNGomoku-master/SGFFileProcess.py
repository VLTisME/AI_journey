
"""Author: zheng.tq@bankcomm.com"""
# author as above. read from .sgf real play data

import time
import os

class SGFflie():
    def __init__(self):
        """
        初始化：
        POS：棋盘坐标的对应字母顺序
        """
        self.POS = 'abcdefghijklmno'

    def openfile(self, filepath):
        """打开文件,读取棋谱"""
        f = open(filepath, 'r', newline='', encoding='ISO-8859-1')
        data = f.read()

        #分割数据
        effective_data = data.split(';')
        s = effective_data[2:-1]

        board = []
        step = 0
        for point in s:
            x = self.POS.find(point[2])
            y = self.POS.find(point[3])
            color = step % 2
            step += 1
            board.append([x, y, color, step])

        f.close()

        return board

    def createTraindataFromqipu(self, path, color=0):
        """将棋谱中的数据生成神经网络训练需要的数据"""
        qipu = self.openfile(path)

        bla = qipu[::2]
        whi = qipu[1::2]
        bla_step = len(bla)
        whi_step = len(whi)

        train_x = []
        train_y = []

        if color == 0:
            temp_x = [0.0 for i in range(225)]
            for index in range(bla_step):
                _x = [0.0 for i in range(225)]
                _y = [0.0 for i in range(225)]
                if index == 0:
                    lx = []
                    lx.append(_x)
                    train_x.append(_x)
                    _y[bla[index][0]*15 + bla[index][1]] = 2.0
                    ly = []
                    ly.append(_y)
                    train_y.append(_y)
                else:
                    _x = temp_x.copy()
                    lx = []
                    lx.append(_x)
                    train_x.append(_x)
                    _y[bla[index][0] * 15 + bla[index][1]] = 2.0
                    ly = []
                    ly.append(_y)
                    train_y.append(_y)

                temp_x[bla[index][0] * 15 + bla[index][1]] = 2.0
                if index < whi_step:
                    temp_x[whi[index][0] * 15 + whi[index][1]] = 1.0

        return train_x, train_y

    def createTraindata(self, path):
        """生成训练数据"""
        filepath = self.allFileFromDir(path)
        train_x = []
        train_y = []
        for path in filepath:
            x, y = self.createTraindataFromqipu(path)

            train_x.extend(x)
            train_y.extend(y)
        return train_x, train_y

    def allFileFromDir(self, Dirpath):
        """获取文件夹中所有文件的路径"""
        pathDir = os.listdir(Dirpath)
        pathfile = []
        for allDir in pathDir:
            child = os.path.join('%s%s' % (Dirpath, allDir))
            pathfile.append(child)
        return pathfile


