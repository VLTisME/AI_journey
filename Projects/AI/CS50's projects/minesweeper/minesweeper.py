import itertools
import random


class Minesweeper():

    def __init__(self, height=8, width=8, mines=8):

        self.height = height
        self.width = width
        self.mines = set()

        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        self.mines_found = set()

    def print(self):
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        count = 0

        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        return self.mines_found == self.mines


class Sentence():

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        if len(self.cells) == self.count:
            return self.cells
        return set()

    def known_safes(self):
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    
    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def get_neighbors(self, cell):
        i, j = cell
        neighbors = set()
        for x in range(max(0, i - 1), min(self.height, i + 2)):
            for y in range(max(0, j - 1), min(self.width, j + 2)):
                neighbors.add((x, y))
        neighbors.remove(cell)
        return neighbors

    def add_knowledge(self, cell, count):

        # Mark the cell as a move that has been made
        self.moves_made.add(cell)

        # Mark the cell as safe
        self.mark_safe(cell)

        self.knowledge.append(Sentence(self.get_neighbors(cell), count))


        did_sth = True
        while did_sth:
            did_sth = False

            # Infer new safe cells and mines
            for sentence in self.knowledge:
                for safe_cell in sentence.known_safes().copy():
                    if safe_cell not in self.safes:
                       self.mark_safe(safe_cell)
                       did_sth = True
                for mine_cell in sentence.known_mines().copy():
                    if mine_cell not in self.mines:
                       self.mark_mine(mine_cell)
                       did_sth = True
            
            for mine in self.mines:
                for sentence in self.knowledge:
                    if mine in sentence.cells.copy():
                       sentence.mark_mine(mine)
                       did_sth = True
            for safe in self.safes:
                for sentence in self.knowledge:
                    if safe in sentence.cells.copy():
                       sentence.mark_safe(safe)
                       did_sth = True

            # Remove empty sentences
            self.knowledge = [sentence for sentence in self.knowledge if sentence.cells]

            # Infer new sentences from existing sentences
            for first_sentence in self.knowledge.copy():
                for second_sentence in self.knowledge.copy():
                    if first_sentence.cells < second_sentence.cells:
                        self.knowledge.append(Sentence(second_sentence.cells - first_sentence.cells, second_sentence.count - first_sentence.count))
                        did_sth = True
                    elif second_sentence.cells < first_sentence.cells:
                        self.knowledge.append(Sentence(first_sentence.cells - second_sentence.cells, first_sentence.count - second_sentence.count))
                        did_sth = True

    def make_safe_move(self):
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell
        return None

    def make_random_move(self):
        for i in range(self.height):
            for j in range(self.width):
                cell = i, j
                if cell not in self.moves_made and cell not in self.mines:
                    return cell
        return None

"""
Những gì học được từ Project này:
1/ Copy a = b với a, b là số thì được. Chứ còn list hay set các kiểu thì phải import copy rồi dùng copy.deepcopy()
chứ không là nó sẽ tham chiếu đến cùng 1 vùng nhớ (tức là thay đổi thằng này thì thằng kia cũng thay đổi theo)
2/ ví dụ như đang ở class X mà có một cái list chứa object của class Y thì nếu ở cả hai class đều có 
hàm tên là abc thì khi gọi hàm abc() trong cái list (ở class X mà dùng object của class Y) thì nó sẽ gọi hàm của class Y chứ không phải X
(ví dụ như ở dòng 127 nó sẽ gọi hàm mark_safe của class Sentence chứ không phải của class MinesweeperAI)
3/ Nếu ở class X mình gọi hàm abc() nào đó mà trong class X không có hàm abc() thì nó sẽ tìm hàm abc() từ class khác
hoặc ở các file khác được import vào hoặc từ global function. Còn nếu ở class X mà có hàm abc() thì nó sẽ gọi hàm abc() của class X
cho dù có hàm abc() ở class khác cũng không ảnh hưởng. Hoặc ở class X không có hàm abc() trong khi hàm Y có thì có thể
gọi kiểu Y.abc() để gọi hàm abc() của class Y cũng được?

Note:
2/ và 3/ mình chưa thực sự hiểu kĩ lắm, cần phải thử nghiệm thêm
"""