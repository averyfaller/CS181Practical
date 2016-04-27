# Imports
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import operator

from SwingyMonkey import SwingyMonkey
from scipy import stats

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
        self.last_action = None
        self.last_reward = 0
        self.gravity = 0

        # For histogram plots so we can create better buckets
        self.vertical_hist = []
        self.horizontal_hist = []

        # This is the learning rate applied to determine to what extent
        # the new information will override the previous information
        self.learning_rate = .9

        # This is the discount factor which we will use to determine the importance of
        # future rewards
        self.discount_factor = .6
        self.q_scores = {}

        # self.vertical_bins = [-400, -200, -150, -100, -50, -25, 0, 25, 35, 45, 55, 65, 75, 85, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 225, 250, 300, 400]
        self.vertical_bins = np.arange(-400, 400, 25)
        self.horizontal_bins = np.arange(-200, 600, 100)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = 0
        self.gravity = 0

        print '-------------------'


    def set_gravity(self, prev_state, next_state):
        self.gravity = prev_state['monkey']['vel'] - next_state['monkey']['vel']

    def should_set_gravity(self):
        return self.gravity == 0

    def state_action_to_array(self, state, action):
        vertical_dist = state['monkey']['bot'] - state['tree']['bot']
        horizontal_dist = state['tree']['dist']

        self.vertical_hist.append(vertical_dist)
        self.horizontal_hist.append(horizontal_dist)

        vertical_bin = np.digitize(vertical_dist, self.vertical_bins)
        horizontal_bin = np.digitize(horizontal_dist, self.horizontal_bins)

        arr = np.array([
              vertical_bin,
              horizontal_bin,
              self.gravity,
              action])

        return arr

    def get_key_from_state_action(self, state, action):
        arr = self.state_action_to_array(state, action)
        return "_".join(map(str, arr))


    def get_Q_score(self, state, action):
        key = self.get_key_from_state_action(state, action)

        if self.q_scores.has_key(key):
            return self.q_scores[key]
        else:
            return 0

    def set_Q_score(self, state, action, q):
        key = self.get_key_from_state_action(state, action)
        self.q_scores[key] = q

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        if state is None or self.last_action is None:
            self.last_state  = state
            new_action = 0
        else:
            if self.should_set_gravity():
                new_action = 0
                if self.last_action == 0 and new_action == 0:
                    self.set_gravity(self.last_state, state)

            prev_Q = self.get_Q_score(self.last_state, self.last_action)
            swing_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 0) - prev_Q)
            jump_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 1) - prev_Q)

            # set new_action to 0 if swing_Q > jump_Q, save best_Q
            new_action, best_Q = max(enumerate([swing_Q, jump_Q]), key=operator.itemgetter(1))

            # Update the Q score for the last state and action
            self.set_Q_score(self.last_state, self.last_action, best_Q)

        print state

        # Update last state and last action for next iteration
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