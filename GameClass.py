import pygame
from PlayerClass import PlayerClass
from FoodClass import FoodClass

class Game_Logic_Core:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('Edureka Snake RL Demo')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        self.bg = pygame.image.load("img/background_main.png")
        self.crash = False
        self.player = PlayerClass(self)
        self.food = FoodClass()
        self.score = 0

if __name__== "__main__":
  main()
