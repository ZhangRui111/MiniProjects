import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        # else:
        #     q_target = r  # next state is terminal
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),  # initialize all values to 0.
                    index=self.q_table.columns,
                    name=state,
                )
            )


class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.at[s, a] = (r, s_)

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.ix[s].dropna().index)    # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.ix[s, a]
        return r, s_
