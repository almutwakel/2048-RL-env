import gym
from gym import spaces
import random
import numpy as np


class Game:
    def __init__(self):
        # Define some colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)

        # self.font = pygame.font.Font('freesansbold.ttf', 32)

        self.texts = []
        # for n in range(0, 12):
        #    text = self.font.render(str(2**n), True, self.BLACK)
        #
        #     self.texts.append(text)

        # This sets the WIDTH and HEIGHT of each grid location
        self.WIDTH = 100
        self.HEIGHT = 100

        # This sets the margin between each cell
        self.MARGIN = 5

        # Create a 2 dimensional array. A two dimensional
        # array is simply a list of lists.
        self.grid = []
        for row in range(4):
            # Add an empty array that will hold each cell
            # in this row
            self.grid.append([])
            for column in range(4):
                self.grid[row].append(0)  # Append a cell

        # Set the HEIGHT and WIDTH of the screen
        self.WINDOW_SIZE = [4*self.WIDTH + 5*self.MARGIN, 5*self.HEIGHT + 6*self.MARGIN]

        # Set title of screen

        # Loop until the user clicks the close button.
        self.done = False

        # Used to manage how fast the screen updates

        # env stuff
        self.action = 0
        self.score = 0

    def add_new_tile(self):
        """
        Adds a new tile in an empty spot in the game board
        """
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        while self.grid[row][col] != 0:
            row = random.randint(0, 3)
            col = random.randint(0, 3)
        self.grid[row][col] = 1

    def check_game_over(self):
        """
        Checks if the game is over
        """
        over = True
        for row in self.grid:
            for col in row:
                if col == 0:
                    over = False
                if col == 11:
                    over = True
                    break

        if over:
            self.done = True
            print("Game over")
            print("Final score:", self.score)

    def start(self):
        self.grid = []
        for row in range(4):
            # Add an empty array that will hold each cell
            # in this row
            self.grid.append([0, 0, 0, 0])
        self.add_new_tile()
        self.add_new_tile()

    def act(self):

        def compress(mat):
            # bool variable to determine
            # any change happened or not
            changed = False

            # empty grid
            new_mat = []

            # with all cells empty
            for i in range(4):
                new_mat.append([0] * 4)

            # here we will shift entries
            # of each cell to it's extreme
            # left row by row
            # loop to traverse rows
            for i in range(4):
                pos = 0

                # loop to traverse each column
                # in respective row
                for j in range(4):
                    if (mat[i][j] != 0):
                        # if cell is non empty then
                        # we will shift it's number to
                        # previous empty cell in that row
                        # denoted by pos variable
                        new_mat[i][pos] = mat[i][j]

                        if (j != pos):
                            changed = True
                        pos += 1

            # returning new compressed matrix
            # and the flag variable.
            return new_mat, changed

        # function to merge the cells
        # in matrix after compressing
        def merge(mat):
            changed = False

            for i in range(4):
                for j in range(3):

                    # if current cell has same value as
                    # next cell in the row and they
                    # are non empty then
                    if (mat[i][j] == mat[i][j + 1] and mat[i][j] != 0):
                        # double current cell value and
                        # empty the next cell
                        mat[i][j] += 1
                        self.score += 2 ** mat[i][j]
                        mat[i][j + 1] = 0

                        # make bool variable True indicating
                        # the new grid after merging is
                        # different.
                        changed = True

            return mat, changed

        # function to reverse the matrix
        # maens reversing the content of
        # each row (reversing the sequence)
        def reverse(mat):
            new_mat = []
            for i in range(4):
                new_mat.append([])
                for j in range(4):
                    new_mat[i].append(mat[i][3 - j])
            return new_mat

        # function to get the transpose
        # of matrix means interchanging
        # rows and column
        def transpose(mat):
            new_mat = []
            for i in range(4):
                new_mat.append([])
                for j in range(4):
                    new_mat[i].append(mat[j][i])
            return new_mat

        # function to update the matrix
        # if we move / swipe left
        def move_left(grid):
            # first compress the grid
            new_grid, changed1 = compress(grid)

            # then merge the cells.
            new_grid, changed2 = merge(new_grid)

            changed = changed1 or changed2

            # again compress after merging.
            new_grid, temp = compress(new_grid)

            # return new matrix and bool changed
            # telling whether the grid is same
            # or different
            return new_grid, changed

        # function to update the matrix
        # if we move / swipe right
        def move_right(grid):
            # to move right we just reverse
            # the matrix
            new_grid = reverse(grid)

            # then move left
            new_grid, changed = move_left(new_grid)

            # then again reverse matrix will
            # give us desired result
            new_grid = reverse(new_grid)
            return new_grid, changed

        # function to update the matrix
        # if we move / swipe up
        def move_up(grid):
            # to move up we just take
            # transpose of matrix
            new_grid = transpose(grid)

            # then move left (calling all
            # included functions) then
            new_grid, changed = move_left(new_grid)

            # again take transpose will give
            # desired results
            new_grid = transpose(new_grid)
            return new_grid, changed

        # function to update the matrix
        # if we move / swipe down
        def move_down(grid):
            # to move down we take transpose
            new_grid = transpose(grid)

            # move right and then again
            new_grid, changed = move_right(new_grid)

            # take transpose will give desired
            # results.
            new_grid = transpose(new_grid)
            return new_grid, changed

        if self.action == 1:
            new_grid, changed = move_up(self.grid)
            self.grid = new_grid
            self.check_game_over()
            if changed:
                self.add_new_tile()
        elif self.action == 2:
            new_grid, changed = move_down(self.grid)
            self.grid = new_grid
            self.check_game_over()
            if changed:
                self.add_new_tile()
        elif self.action == 3:
            new_grid, changed = move_left(self.grid)
            self.grid = new_grid
            self.check_game_over()
            if changed:
                self.add_new_tile()
        elif self.action == 4:
            new_grid, changed = move_right(self.grid)
            self.grid = new_grid
            self.check_game_over()
            if changed:
                self.add_new_tile()
        self.action = 0


class Env2048(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env2048, self).__init__()
        # Define action and observation space
        # 1 = up 2 = down 3 = left 4 = right
        self.action_space = spaces.Discrete(4)
        # 4x4 grid
        self.observation_space = spaces.Box(low=0, high=11, shape=(16,), dtype=int)
        self.play = Game()
        self.state = None
        self.reset()
        # self.render()

    def step(self, action):
        # Execute one time step within the environment
        prev_score = self.play.score
        self.play.action = action
        self.play.act()
        self.state = sum(self.play.grid, [])
        if self.play.done:
            done = True
            reward = 0
        else:
            done = False
            reward = self.play.score - prev_score
        # self.render()
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # super().reset()
        self.play.start()
        self.state = sum(self.play.grid, [])
        return np.array(self.state)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        from gym.envs.classic_control import rendering
        print(self.play.grid)

    def close(self):
        pass


# env = Env2048()
# env.step(1)
# env.step(1)
# env.step(2)
