# Imports
import numpy as np
import numpy.random as npr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from SwingyMonkey import SwingyMonkey
from scipy import stats
import operator

from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = 0
        self.gravity = 0
        
        self.num_vars = 4
        
        # This is the learning rate applied to determine to what extent
        # the new information will override the previous information
        self.learning_rate = .9
        
        # This is the discount factor which we will use to determine the importance of 
        # future rewards
        self.discount_factor = .6

        # The matrix where we will store Q-scores for (state, action) tuples
        self.epoch_X = np.zeros(self.num_vars)
        self.X = np.zeros(self.num_vars)
        self.y = np.zeros(1)
        
        ##############################
        ### CHOOSE REGRESSION TYPE ###
        ##############################
        # Random Forest Regressor
        self.rf = RandomForestRegressor(n_estimators=50)
        
        # Linear Regression
        #self.rf = LinearRegression()
        
        # Gradient Boosting Regressor
        #self.rf = GradientBoostingRegressor(loss='quantile')
        
        # Neural Network with Scaling
        #self.rf = Pipeline([
        #    ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
        #    ('neural network', Regressor(layers=[
        #        Layer("Sigmoid", units=10),
        #        Layer("Sigmoid", units=10),           
        #        Layer("Linear")],
        #    learning_rate=0.02,
        #    n_iter=20))])

        
        self.epsilon = 1.0
        self.fitted = False

        self.vertical_bins = np.arange(-400, 400, 25)
        self.horizontal_bins = np.arange(-200, 600, 100)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = 0
        
        self.epsilon = self.epsilon * .9

        # Remove the first row
        self.epoch_X = self.epoch_X[1:,]
        self.epoch_X[:,-2] = self.gravity
        self.gravity = 0

        self.X = np.vstack((self.X, self.epoch_X))
        #print '-------------------'

        self.epoch_X = np.zeros(self.num_vars)
        
        # Refit at the start of each epoch
        if len(self.X) > 0:
            self.rf.fit(self.X, self.y)
            score = self.rf.score(self.X, self.y)
            self.fitted = True
            print "Refitting for new epoch, score: %f" % score

    def state_action_to_array(self, state, action):
        if state is not None:
            vertical_dist = state['monkey']['bot'] - state['tree']['bot']
            horizontal_dist = state['tree']['dist']

            vertical_bin = np.digitize(vertical_dist, self.vertical_bins)
            horizontal_bin = np.digitize(horizontal_dist, self.horizontal_bins)
            
            arr = np.array([
                  vertical_bin,
                  horizontal_bin,
                  self.gravity,
                  action])
            
            return arr
        return 0

    def get_Q_score(self, state, action):
        if state is None or action is None or not self.fitted:
            return 0
        
        return self.rf.predict(np.array(self.state_action_to_array(state, action)).reshape(1, -1))
    
    def set_Q_score(self, state, action, q):
        arr = self.state_action_to_array(state, action)
        if arr is not None:
            self.epoch_X = np.vstack((self.epoch_X, arr))
            self.y = np.append(self.y, q)

    def set_gravity(self, prev_state, next_state):
        self.gravity = prev_state['monkey']['vel'] - next_state['monkey']['vel']

    def should_set_gravity(self):
        return self.gravity == 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        if state is None or self.last_action is None:
            self.last_state = state
            new_action = 0
        else:
            if self.should_set_gravity():
                new_action = 0
                if self.last_action == 0 and new_action == 0:
                    self.set_gravity(self.last_state, state)

            prev_Q = self.get_Q_score(self.last_state, self.last_action)
            swing_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 0) - prev_Q)
            jump_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 1) - prev_Q)

            #print "SWING %f, JUMP %f" % (swing_Q, jump_Q)
            # set new_action to 0 if swing_Q > jump_Q, save best_Q
            new_action, best_Q = max(enumerate([swing_Q, jump_Q]), key=operator.itemgetter(1))
            
            if (swing_Q == jump_Q) or (np.random.rand() < self.epsilon):
                new_action = (np.random.rand() < .08) * 1

            # Update the Q score for the last state and action
            self.set_Q_score(self.last_state, self.last_action, best_Q)

        # Update last state and last action for next iteration
        self.last_state  = state
        self.last_action = new_action

        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100, r_iters = 10):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    print "Running %d games" % iters
    
    for ii in range(iters):
        # Make a new monkey object.
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
    run_games(agent, hist, 100, 1)
    
    # Save history. 
    np.save('hist',np.array(hist))
    
    # Print max score
    print "Max score was: %f" % max(hist)
    print "Total Score was %d" % np.sum(hist)