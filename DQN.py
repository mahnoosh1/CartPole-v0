from __future__ import print_function, division, absolute_import

import os
import sys
import numpy as np
import datetime
import random
import math
from collections import namedtuple, deque
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym_miniworld


###################################
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.00
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.batch_size = 64

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, kernel_initializer="he_uniform", activation="relu"))
        model.add(Dense(24, kernel_initializer="he_uniform", activation="relu"))
        model.add(Dense(self.action_size, activation="linear", kernel_initializer="he_uniform"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() < self.epsilon:
            print("random action is " + str(np.random.choice(np.arange(self.action_size))))
            return np.random.choice(np.arange(self.action_size))
        else:
            act_values = self.model.predict(state)[0]
            print("act values "+str(act_values)+"   "+str(np.argmax(act_values)))
            # print("action is " + str(np.argmax(act_values)))

            return np.argmax(act_values)


###################################

if __name__ == "__main__":
    EPISODES = 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    scores = []
    batch_size = 128
    done = False

    for e in range(1, EPISODES + 1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        render_start = False
        render_stop = False
        for time_p in range(500):
            # if render_start:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            agent.remember(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                agent.update_target_model()
                score = score if score == 500 else score + 100
                scores.append(score)
                print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(e, EPISODES, score, agent.epsilon))
                break
        if render_stop:
            env.close()
