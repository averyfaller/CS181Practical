# Imports
import numpy as np
import numpy.random as npr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey
from scipy import stats

from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# get histograms - NC
# try different bins - EW
# talk about epsilon greedy approach - EW
# talk about neural nets / random forest - NC / AF
# talk about dictionary - EW

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.penultimate_state = None
        self.last_action = None
        self.last_reward = 0
        self.num_vars = 6
        self.gravity = 0

        self.vertical_hist = []
        self.horizontal_hist = []

        # This is the learning rate applied to determine to what extent
        # the new information will override the previous information
        self.learning_rate = .9

        # This is the discount factor which we will use to determine the importance of
        # future rewards
        self.discount_factor = .6

        # The matrix where we will store Q-scores for (state, action) tuples
        self.X = np.zeros(self.num_vars)
        self.y = np.zeros(1)

        self.q_scores = {}

        self.epsilon = 1.0

        self.epoch_gravity = np.array(1)

    def reset(self):
        self.last_state  = None
        self.penultimate_state = None
        self.last_action = None
        self.last_reward = 0
        self.train = False
        self.gravity = 0

        self.epsilon = self.epsilon * .5

        print '-------------------'

        self.epoch_gravity = np.array(1)

    def set_gravity(self, prev_state, next_state):
        self.gravity = prev_state['monkey']['vel'] - next_state['monkey']['vel']

    def state_action_to_array(self, state, action):
        if state is not None:
            vertical_dist = state['monkey']['bot'] - state['tree']['bot']
            horizontal_dist = state['tree']['dist']

            self.vertical_hist.append(vertical_dist)
            self.horizontal_hist.append(horizontal_dist)

            # vertical_bins = [-400, -200, -150, -100, -50, -25, 0, 25, 35, 45, 55, 65, 75, 85, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 225, 250, 300, 400]
            vertical_bins = np.arange(-400, 400, 25)
            horizontal_bins = np.arange(-200, 600, 100)

            vertical_bin = np.digitize(vertical_dist, vertical_bins)
            horizontal_bin = np.digitize(horizontal_dist, horizontal_bins)

            arr = np.array([
                  vertical_bin,
                  horizontal_bin,
                  self.gravity,
                  action])

            return arr
        return None

    def get_Q_score(self, state, action):
        if state is None or action is None:
            return -1

        arr = self.state_action_to_array(state, action)
        key = "_".join(map(str, arr))

        if self.q_scores.has_key(key):
            return self.q_scores[key]
        else:
            return 0

    def set_Q_score(self, state, action, q):
        arr = self.state_action_to_array(state, action)
        if arr is not None:
            key = "_".join(map(str, arr))
            self.q_scores[key] = q

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        prev_Q = self.get_Q_score(self.last_state, self.last_action)

        swing_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 0) - prev_Q)

        jump_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 1) - prev_Q)

        # print "SWING %f, JUMP %f" % (swing_Q, jump_Q)

        print state

        # Pick the better Q score from the possible actions at s_{t+1}
        action_Qs = [swing_Q, jump_Q]
        best_action = 0
        if jump_Q > swing_Q:
            best_action = 1

        if self.train or jump_Q == swing_Q: # or npr.rand() < self.epsilon:
            new_action = (npr.rand() < 0.1) * 1
        else:
            new_action = best_action

        best_Q = action_Qs[new_action]

        if self.gravity == 0:
            new_action = 0
            if self.last_action == 0 and new_action == 0:
                self.set_gravity(self.last_state, state)

        # Update the Q score for the last state and action
        self.set_Q_score(self.last_state, self.last_action, best_Q)

        self.penultimate_state = self.last_state
        self.last_state  = state
        self.last_action = new_action

        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

def run_games(learner, hist, iters = 1000, t_len = 100, r_iters = 10):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    print "Running %d games" % iters

    for ii in range(iters):
        # Make a new monkey object.
        if ii < r_iters:
            learner.train = True
            print "Running a training epoch"

        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d, Max Score %d" % (ii, max(hist)) ,       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = [0]

    # Run games.
    run_games(agent, hist, 100, 1, 1)

    #print "Num states: %d" % len(agent.Q)

    # Save history.
    np.save('hist',np.array(hist))

    # Print max score
    print "Max score was: %f" % max(hist)
    print "Total Score was %d" % np.sum(hist)