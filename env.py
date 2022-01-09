import gym
from gym import spaces
import random
import numpy as np
import display
from gym.envs.classic_control import rendering
import time
import pyglet


class Game:
    def __init__(self):
        # Define some colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.colorlist = [(238, 228, 218),
                          (238, 228, 218),
                          (242, 177, 121),
                          (245, 149, 99),
                          (246, 124, 95),
                          (246, 94, 59),
                          (237, 207, 114),
                          (237, 204, 97),
                          (237, 200, 80),
                          (237, 197, 63),
                          (237, 194, 46)]
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

        self.done = False

        # env stuff
        self.action = -1
        self.score = 0

    def compress(self, mat):
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
    def merge(self, mat):
        changed = False
        score_delta = 0
        for i in range(4):
            for j in range(3):

                # if current cell has same value as
                # next cell in the row and they
                # are non empty then
                if (mat[i][j] == mat[i][j + 1] and mat[i][j] != 0):
                    # double current cell value and
                    # empty the next cell
                    mat[i][j] += 1
                    score_delta = 2 ** mat[i][j]
                    mat[i][j + 1] = 0

                    # make bool variable True indicating
                    # the new grid after merging is
                    # different.
                    changed = True

        return mat, changed, score_delta

    # function to reverse the matrix
    # maens reversing the content of
    # each row (reversing the sequence)
    def reverse(self, mat):
        new_mat = []
        for i in range(4):
            new_mat.append([])
            for j in range(4):
                new_mat[i].append(mat[i][3 - j])
        return new_mat

    # function to get the transpose
    # of matrix means interchanging
    # rows and column
    def transpose(self, mat):
        new_mat = []
        for i in range(4):
            new_mat.append([])
            for j in range(4):
                new_mat[i].append(mat[j][i])
        return new_mat

    # function to update the matrix
    # if we move / swipe left
    def move_left(self, grid):
        # first compress the grid
        new_grid, changed1 = self.compress(grid)

        # then merge the cells.
        new_grid, changed2, score_delta = self.merge(new_grid)

        changed = changed1 or changed2

        # again compress after merging.
        new_grid, temp = self.compress(new_grid)

        # return new matrix and bool changed
        # telling whether the grid is same
        # or different
        return new_grid, changed, score_delta

    # function to update the matrix
    # if we move / swipe right
    def move_right(self, grid):
        # to move right we just reverse
        # the matrix
        new_grid = self.reverse(grid)

        # then move left
        new_grid, changed, scoredelta = self.move_left(new_grid)

        # then again reverse matrix will
        # give us desired result
        new_grid = self.reverse(new_grid)
        return new_grid, changed, scoredelta

    # function to update the matrix
    # if we move / swipe up
    def move_up(self, grid):
        # to move up we just take
        # transpose of matrix
        new_grid = self.transpose(grid)

        # then move left (calling all
        # included functions) then
        new_grid, changed, score_delta = self.move_left(new_grid)

        # again take transpose will give
        # desired results
        new_grid = self.transpose(new_grid)
        return new_grid, changed, score_delta

    # function to update the matrix
    # if we move / swipe down
    def move_down(self, grid):
        # to move down we take transpose
        new_grid = self.transpose(grid)

        # move right and then again
        new_grid, changed, score_delta = self.move_right(new_grid)

        # take transpose will give desired
        # results.
        new_grid = self.transpose(new_grid)
        return new_grid, changed, score_delta


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

    def check_valid_move(self, move):
        copy = np.copy(self.grid)
        if move == 0:
            new_grid, changed, _ = self.move_up(copy)
        elif move == 1:
            new_grid, changed, _ = self.move_down(copy)
        elif move == 2:
            new_grid, changed, _ = self.move_left(copy)
        elif move == 3:
            new_grid, changed, _ = self.move_right(copy)
        else:
            return Exception
        return changed

    def check_game_over(self):
        """
        Checks if the game is over
        """
        over = True
        for row in range(4):
            for col in range(4):
                if self.grid[row][col] == 0:
                    over = False
                if over:
                    for move in range(4):
                        if self.check_valid_move(move):
                            over = False
                if self.grid[row][col] == 11:
                    self.done = True
                    print("2048! Game over")
                    print("Final score:", self.score)
                    return

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
        print("acted", self.action)
        if self.action == 0:
            new_grid, changed, score_delta = self.move_up(self.grid)
        elif self.action == 1:
            new_grid, changed, score_delta = self.move_down(self.grid)
        elif self.action == 2:
            new_grid, changed, score_delta = self.move_left(self.grid)
        elif self.action == 3:
            new_grid, changed, score_delta = self.move_right(self.grid)
        else:
            print(Exception, "Invalid action")
            return

        self.grid = new_grid
        if changed:
            self.add_new_tile()
        self.score += score_delta
        self.check_game_over()
        self.action = -1
        return score_delta


class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()


class Env2048(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, conv=False):
        super(Env2048, self).__init__()
        # Define action and observation space
        # 0 = up 1 = down 2 = left 3 = right
        self.action_space = spaces.Discrete(4)
        # 4x4 grid
        self.conv = conv
        if self.conv:
            self.observation_space = spaces.Box(low=0, high=11, shape=(4, 4, 1), dtype=int)
        else:
            self.observation_space = spaces.Box(low=0, high=11, shape=(16,), dtype=int)
        self.play = Game()
        self.state = None
        screen_width = 400
        screen_height = 500
        self.viewer = rendering.Viewer(screen_width, screen_height)
        self.reset()
        # self.render()

    def step(self, action):
        # Execute one time step within the environment
        prev_score = self.play.score
        self.play.action = action
        reward = self.play.act()
        if self.conv:
            self.state = self.play.grid
        else:
            self.state = sum(self.play.grid, [])
        return self.state, reward, self.play.done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # super().reset()
        self.play.start()
        self.play.done = False
        self.play.score = 0
        if self.conv:
            self.state = self.play.grid
        else:
            self.state = sum(self.play.grid, [])
        return self.state

    def render(self, mode='human', close=False):
        self.viewer.geoms = []
        background = rendering.FilledPolygon([(0, 0), (0, 500), (400, 500), (400, 0)])
        self.viewer.add_geom(background)
        label = pyglet.text.Label(str(self.play.score), font_size=36,
                                  x=200, y=450, anchor_x='left', anchor_y='bottom',
                                  color=(255, 123, 255, 255))

        label.draw()

        # 238, 228, 218
        for row in range(4):
            for col in range(4):
                if self.conv:
                    tileval = self.state[row][col]
                else:
                    tileval = self.state[row * 4 + col]

                if tileval > 0:
                    tile = rendering.FilledPolygon([(10 + 100 * col, 10 + 100 * row), (10 + 100 * col, 90 + 100 * row),
                                                    (90 + 100 * col, 90 + 100 * row), (90 + 100 * col, 10 + 100 * row)])
                    tile.set_color(self.play.colorlist[tileval][0]/255, self.play.colorlist[tileval][1]/255,
                                   self.play.colorlist[tileval][2]/255)
                    tile_transform = rendering.Transform()
                    tile.add_attr(tile_transform)
                    self.viewer.add_geom(tile)
                    label = pyglet.text.Label(str(2 ** tileval), font_size=18,
                                              x=50 + 100 * col, y=50 + 100 * row, anchor_y='center', anchor_x='center',
                                              color=(0, 0, 0, 255))
                    label.draw()
                    self.viewer.add_geom(DrawText(label))

        if self.state is None:
            return None

        # Text
        label = pyglet.text.Label(str(self.play.score), font_size=18,
                                  x=200, y=450, anchor_y='center', anchor_x='center',
                                  color=(255, 255, 255, 255))
        label.draw()
        self.viewer.add_geom(DrawText(label))
        # time.sleep(0.01)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        self.viewer.geoms = []

