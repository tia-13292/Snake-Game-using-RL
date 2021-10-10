import pygame
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from GameFunctions import *
import json

class FoodClass(object):

    def __init__(self):
        self.config = json.loads(open("config.txt").read())
        self.image = pygame.image.load(self.config['food_image'])
        self.x_dimension_food = int(self.config['x_dimension_food'])
        self.y_dimension_food = int(self.config['y_dimension_food'])

    def position_food(self, game, player):
        x_rand = randint(25, game.game_width - 40)
        self.x_dimension_food = x_rand - x_rand % 20
        y_rand = randint(25, game.game_height - 40)
        self.y_dimension_food = y_rand - y_rand % 25
        if [self.x_dimension_food, self.y_dimension_food] not in player.position:
            return self.x_dimension_food, self.y_dimension_food
        else:
            self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()

if __name__== "__main__":
  main()
