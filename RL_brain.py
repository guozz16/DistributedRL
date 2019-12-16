"""
This part of code is the Q learning brain, which is a brain of the agent.
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, \
        e_greedy=0.9, q_table=None):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        if q_table is None:
            self.q_table = pd.DataFrame(columns=self.actions+['MEMO'], dtype=np.float64)
        else:
            self.q_table = pd.read_csv(q_table,header=0,index_col=0)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        self.q_table['MEMO'][observation] += 1 # MEMO state count plus 1
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :self.actions[-1]]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def info_learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        freq = self.q_table.loc[s, 'MEMO']/self.q_table['MEMO'].sum()
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict) * (-np.log2(freq))/100 # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            print('aha')
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*(len(self.actions)+1),
                    index=self.q_table.columns,
                    name=state,
                )
            )