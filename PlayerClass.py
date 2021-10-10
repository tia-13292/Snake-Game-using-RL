import pygame
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from GameFunctions import *
import json
class PlayerClass(object):

    def __init__(self, game):
        self.config = json.loads(open("config.txt").read())
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.snake_x = x - x % 20
        self.snake_y = y - y % 20
        self.snake_position = []
        self.snake_position.append([self.snake_x, self.snake_y])
        self.snake_food = 1
        self.snake_has_eaten = False
        self.snake_body_image = pygame.image.load(self.config['snake_body_image'])
        self.snake_x_change = int(self.config['movement'])
        self.snake_y_change = 0

    def update_position(self, x, y):
        if self.snake_position[-1][0] != x or self.snake_position[-1][1] != y:
            if self.snake_food > 1:
                for i in range(0, self.snake_food - 1):
                    self.snake_position[i][0], self.position[i][1] = self.position[i + 1]
            self.snake_position[-1][0] = x
            self.snake_position[-1][1] = y

    def do_move(self, move, x, y, game, food,agent):
        move_array = [self.snake_x_change, self.snake_y_change]

        if self.snake_has_eaten:

            self.snake_position.append([self.snake_x, self.snake_x])
            self.snake_has_eaten = False
            self.snake_food = self.snake_food + 1
        if np.array_equal(move ,[1, 0, 0]):
            move_array = self.snake_x_change, self.snake_y_change
        elif np.array_equal(move,[0, 1, 0]) and self.snake_y_change == 0:
            move_array = [0, self.snake_x_change]
        elif np.array_equal(move,[0, 1, 0]) and self.snake_x_change == 0:
            move_array = [-self.snake_y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.snake_y_change == 0:
            move_array = [0, -self.snake_x_change]
        elif np.array_equal(move,[0, 0, 1]) and self.snake_x_change == 0:
            move_array = [self.snake_y_change, 0]
        self.snake_x_change, self.snake_y_change = move_array
        self.snake_x = x + self.snake_x_change
        self.snake_y = y + self.snake_y_change

        if self.snake_x < 20 or self.snake_x > game.game_width-40 or self.snake_y < 20 or self.snake_y > game.game_height-40 or [self.snake_x, self.snake_y] in self.snake_position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.snake_x, self.snake_y)

    def display_player(self, x, y, food, game):
        self.snake_position[-1][0] = x
        self.snake_position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.snake_position[len(self.snake_position) - 1 - i]
                game.gameDisplay.blit(self.snake_body_image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)

if __name__== "__main__":
  main()
