import pygame
from random import randint
from Q_Logic import SnakeAgent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from GameClass import Game_Logic_Core
from PlayerClass import PlayerClass
import sys
import json
from GameFunctions import *
pygame.font.init()



def main(config):
    pygame.init()
    agent = SnakeAgent()
    game_counter = 0
    score_plot = []
    counter_plot =[]
    record = 0
    while game_counter < int(config['total_iteration']):
        Game_Logic_Core_Object = Game_Logic_Core(eval(config['game_grid'])[0], eval(config['game_grid'])[1])
        player_core = Game_Logic_Core_Object.player
        food = Game_Logic_Core_Object.food
        print ("THE DQN is learning how to play SNAKE .......")
        initialize_game(player_core, Game_Logic_Core_Object, food, agent)
        if display_option:
            display(player_core, food, Game_Logic_Core_Object, record)

        while not Game_Logic_Core_Object.crash:
            agent.epsilon = int(config['epsilon']) - game_counter
            state_old = agent.get_game_state(Game_Logic_Core_Object, player_core, food)
            if randint(0, 150) < agent.epsilon:
                final_move = tf.keras.utils.to_categorical(randint(0, 2), num_classes=3)
            else:
                prediction = agent.model.predict(state_old.reshape((1,11)))
                final_move = tf.keras.utils.to_categorical(np.argmax(prediction[0]), num_classes=3)

            player_core.do_move(final_move, player_core.snake_x, player_core.snake_y, Game_Logic_Core_Object, food, agent)
            state_new = agent.get_game_state(Game_Logic_Core_Object, player_core, food)

            reward = agent.reward_init(player_core, Game_Logic_Core_Object.crash)
            agent.minibatch_train(state_old, final_move, reward, state_new, Game_Logic_Core_Object.crash)

            agent.remember_state(state_old, final_move, reward, state_new, Game_Logic_Core_Object.crash)
            record = get_record(Game_Logic_Core_Object.score, record)
            if display_option:
                display(player_core, food, Game_Logic_Core_Object, record)
                pygame.time.wait(speed)

        agent.replay_reset_new(agent.memory)
        game_counter += 1
        if eval(config['print_learning_scores']):
            print('Learning Stage -> ', game_counter, '      Max Score Achived:', Game_Logic_Core_Object.score)
        score_plot.append(Game_Logic_Core_Object.score)
        counter_plot.append(game_counter)
    if eval(config['save_weights']):
        agent.model.save_weights(config['model_name_aftertrain'])
    if eval(config['learning_curve']):
        plot_seaborn(counter_plot, score_plot)


if __name__== "__main__":
    config = json.loads(open("config.txt").read())
    display_option = eval(config['display_option'])
    speed = int(config['game_speed'])
    main(config)
