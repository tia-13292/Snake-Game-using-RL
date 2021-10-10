import pygame
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def eat(player, food, game):
    if player.snake_x == food.x_dimension_food and player.snake_y == food.y_dimension_food:
        food.position_food(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.snake_position[-1][0], player.snake_position[-1][1], player.snake_food, game)
    food.display_food(food.x_dimension_food, food.y_dimension_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent):
    state_init1 = agent.get_game_state(game, player, food)
    action = [1, 0, 0]
    player.do_move(action, player.snake_x, player.snake_y, game, food, agent)
    state_init2 = agent.get_game_state(game, player, food)
    reward1 = agent.reward_init(player, game.crash)
    agent.remember_state(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_reset_new(agent.memory)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()

if __name__== "__main__":
  main()
