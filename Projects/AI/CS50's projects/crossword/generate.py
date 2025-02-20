import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):

        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        remove_words = list()
        for var in self.crossword.variables:
            for word in self.domains[var]:
                if len(word) != var.length:
                    remove_words.append(word)
            for word in remove_words:
                self.domains[var].remove(word)
            remove_words.clear()

    def revise(self, x, y):
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        i, j = overlap
        remove_words = list()
        for word in self.domains[x]:
            if not any(word[i] == word2[j] for word2 in self.domains[y]):
                remove_words.append(word)
                revised = True
        for word in remove_words:
            self.domains[x].remove(word)
        return revised

    def ac3(self, arcs=None):
        if arcs is None:
            arcs = [(x, y) for x in self.crossword.variables for y in self.crossword.neighbors(x)]
        while arcs:
            x, y = arcs.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))
        return True

    def assignment_complete(self, assignment):
        for var in self.domains.keys():
            if var not in assignment.keys():
                return False
        return True

    def consistent(self, assignment):
        for x in assignment.keys():
            if x.length != len(assignment[x]):
                return False
            for y in self.crossword.neighbors(x):
                if (y in assignment.keys()):
                    i, j = self.crossword.overlaps[x, y]
                    if assignment[x][i] != assignment[y][j]:
                        return False
        if (len(set(assignment.values())) != len(assignment.values())):
            return False
        return True

    def order_domain_values(self, var, assignment):
        rank_dict = list()
        for word in self.domains[var]:
            eliminated = 0
            for var2 in self.crossword.neighbors(var):
                if var2 in assignment.keys():
                    continue
                i, j = self.crossword.overlaps[var, var2]
                for word2 in self.domains[var2]:
                    if word[i] != word2[j]:
                        eliminated += 1
            rank_dict.append((word, eliminated))
        rank_dict.sort(key = lambda x: x[1])
        return list(word for word, _ in rank_dict)
        

    def select_unassigned_variable(self, assignment):
        arr = list()
        for key in self.domains.keys():
            if key not in assignment.keys():
                arr.append((key, len(self.domains[key]), len(self.crossword.neighbors(key))))
        arr.sort(key = lambda x: (x[1], -x[2]))
        return arr[0][0]

    def backtrack(self, assignment):
        if (self.assignment_complete(assignment)):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            assignment[var] = None
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
