import gym
import gym_miniworld
import numpy as np
from collections import namedtuple, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory
import random


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
            return np.random.choice(np.arange(self.action_size))
        else:
            act_values = self.model.predict(state)[0]
            return np.argmax(act_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(batch_size):
            update_input[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            update_target[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + (self.gamma * np.amax(target_val[i]))
        self.model.fit(update_input, target, epochs=1, verbose=0)


if __name__ == "__main__":
    EPISODES = 500
    env = gym.make('MiniWorld-Maze-v0')
    state_size = env.observation_space.shape
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
            agent.replay()
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
