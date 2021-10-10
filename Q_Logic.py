import tensorflow as tf
import random
import numpy as np
import pandas as pd
from operator import add
import json

class SnakeAgent(object):

    def __init__(self):
        self.config = json.loads(open("config.txt").read())
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005  # 0.00005 # 0.0000005
        if eval(self.config['use_pre_trained_model']):
            self.model = self.deep_network(self.config['pre_trained_model_name'])
        else:
            self.model = self.deep_network()
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_game_state(self, game, player, food):

        movement = int(self.config['movement'])
        state = [
            (player.snake_x_change == movement and player.snake_y_change == 0 and ((list(map(add, player.snake_position[-1], [movement, 0])) in player.snake_position) or
            player.snake_position[-1][0] + movement >= (game.game_width - movement))) or (player.snake_x_change == -movement and player.snake_y_change == 0 and ((list(map(add, player.snake_position[-1], [-movement, 0])) in player.snake_position) or
            player.snake_position[-1][0] - movement < movement)) or (player.snake_x_change == 0 and player.snake_y_change == -movement and ((list(map(add, player.snake_position[-1], [0, -movement])) in player.snake_position) or
            player.snake_position[-1][-1] - movement < movement)) or (player.snake_x_change == 0 and player.snake_y_change == movement and ((list(map(add, player.snake_position[-1], [0, movement])) in player.snake_position) or
            player.snake_position[-1][-1] + movement >= (game.game_height-movement))),

            (player.snake_x_change == 0 and player.snake_y_change == -movement and ((list(map(add,player.snake_position[-1],[movement, 0])) in player.snake_position) or
            player.snake_position[ -1][0] + movement > (game.game_width-movement))) or (player.snake_x_change == 0 and player.snake_y_change == movement and ((list(map(add,player.snake_position[-1],
            [-movement,0])) in player.snake_position) or player.snake_position[-1][0] - movement < movement)) or (player.snake_x_change == -movement and player.snake_y_change == 0 and ((list(map(
            add,player.snake_position[-1],[0,-movement])) in player.snake_position) or player.snake_position[-1][-1] - movement < movement)) or (player.snake_x_change == movement and player.snake_y_change == 0 and (
            (list(map(add,player.snake_position[-1],[0,movement])) in player.snake_position) or player.snake_position[-1][
             -1] + movement >= (game.game_height-movement))),

             (player.snake_x_change == 0 and player.snake_y_change == movement and ((list(map(add,player.snake_position[-1],[movement,0])) in player.snake_position) or
             player.snake_position[-1][0] + movement > (game.game_width-movement))) or (player.snake_x_change == 0 and player.snake_y_change == -movement and ((list(map(
             add, player.snake_position[-1],[-movement,0])) in player.snake_position) or player.snake_position[-1][0] - movement < movement)) or (player.snake_x_change == movement and player.snake_y_change == 0 and (
            (list(map(add,player.snake_position[-1],[0,-movement])) in player.snake_position) or player.snake_position[-1][-1] - movement < movement)) or (
            player.snake_x_change == -movement and player.snake_y_change == 0 and ((list(map(add,player.snake_position[-1],[0,movement])) in player.snake_position) or
            player.snake_position[-1][-1] + movement >= (game.game_height-movement))),


            player.snake_x_change == -movement,
            player.snake_x_change == movement,
            player.snake_y_change == -movement,
            player.snake_y_change == movement,
            food.x_dimension_food < player.snake_x,
            food.x_dimension_food > player.snake_x,
            food.y_dimension_food < player.snake_y,
            food.y_dimension_food > player.snake_y
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def reward_init(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -15
            return self.reward
        if player.snake_has_eaten:
            self.reward = 20
        return self.reward

    def deep_network(self, weights=None):
        model = tf.keras.Sequential()
        for numberofLayer in range(int(self.config['TotalDenseLayer'])):
            model.add(tf.keras.layers.Dense(units=int(self.config['TotalDenseLayer']), activation=tf.keras.activations.relu, input_dim=11))
            model.add(tf.keras.layers.Dropout(float(self.config['Dropout'])))

        model.add(tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam((self.learning_rate)))

        if weights:
            model.load_weights(weights)
        return model

    def remember_state(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_reset_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)


    def minibatch_train(self, state, action, reward, next_state, done):

        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
