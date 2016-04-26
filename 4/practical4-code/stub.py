# Imports
import numpy as np
import numpy.random as npr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from SwingyMonkey import SwingyMonkey
from scipy import stats

from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
        
        # This is the learning rate applied to determine to what extent
        # the new information will override the previous information
        self.learning_rate = .1
        
        # This is the discount factor which we will use to determine the importance of 
        # future rewards
        self.discount_factor = .9

        # The matrix where we will store Q-scores for (state, action) tuples
        self.epoch_X = np.zeros(self.num_vars)
        self.X = np.zeros(self.num_vars)
        self.y = np.zeros(1)
        self.rf = RandomForestRegressor(n_estimators=50)
        #self.rf = LinearRegression()
        #self.rf = GradientBoostingRegressor(loss='quantile')
        #self.rf = Pipeline([
        #    ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
        #    ('neural network', Regressor(layers=[
        #        Layer("Sigmoid", units=50),
        #        Layer("Sigmoid", units=50),           
        #        Layer("Linear")],
        #    learning_rate=0.02,
        #    n_iter=20))])

        self.epsilon = 1.0
        
        self.fitted = False
        self.epoch_gravity = np.array(1)

    def reset(self):
        self.last_state  = None
        self.penultimate_state = None
        self.last_action = None
        self.last_reward = 0
        self.train = False
        
        self.epsilon = self.epsilon * .9

        # Remove the first row
        self.epoch_X = self.epoch_X[1:,]
        
        # TODO: conver the epoch_X to a matrix so we can run this line of code:
        self.epoch_X[:,-2] = stats.mode(self.epoch_gravity)[0][0]

        self.X = np.vstack((self.X, self.epoch_X))
        print '-------------------'

        self.epoch_X = np.zeros(self.num_vars)
        
        self.epoch_gravity = np.array(1)
        
        # Refit at the start of each epoch
        if len(self.X) > 0: # and self.last_reward < 5:
            #print self.X[-1], self.y[-1]
            self.rf.fit(self.X, self.y)
            score = self.rf.score(self.X, self.y)
            self.fitted = True
            print "Refitting for new epoch, score: %f" % score

    def state_action_to_array(self, state, action, prev_state, set_epoch_graivty=False):
        if state is not None and prev_state is not None:
            #print state['tree']['top'] - state['tree']['bot']
            #arr = [state['tree']['bot'], state['tree']['top'], state['tree']['dist'], 
            #   state['monkey']['bot'], state['monkey']['top'], state['monkey']['vel'],
            #  action]
            #print state['monkey']['top'] - state['monkey']['bot']
            #state['tree']['top'] - (state['monkey']['top'], 
            #print state['monkey']['top'] - state['monkey']['bot']
            # prev_state['monkey']['vel'] - state['monkey']['vel'], 
            #                   prev_state['tree']['top'],
            
            #                    ((state['tree']['top'] - 100) - (state['monkey']['top'] - 28))**2,
            
            #                    state['monkey']['bot'],
            #        state['tree']['bot'], 
            if set_epoch_graivty and action == 0:
                self.epoch_gravity = np.append(self.epoch_gravity, prev_state['monkey']['vel'] - state['monkey']['vel'])
            
                               
            #       prev_state['monkey']['top'],
            #       prev_state['tree']['top'],
            #       prev_state['tree']['dist'],
            #       prev_state['monkey']['vel'],
                   
                                    
                   #prev_state['monkey']['top']// boxes,
                   #prev_state['tree']['top']// boxes,
                   #prev_state['tree']['dist']// boxes,
                   #prev_state['monkey']['vel']// vel_boxing,
            # np.sqrt(((state['tree']['top'] - 100) - (state['monkey']['top'] - 28))**2 + (state['tree']['dist'])**2) // boxes,
                
           #                   (state['monkey']['top'])**2 // boxes,      
                
            #                   state['tree']['top'] // boxes, 
            
            #                                       
            
            boxes = 3
            vel_boxing = 1
            
            #arr = np.array([
            #       state['monkey']['top'] // boxes, 
            #        state['monkey']['bot'] // boxes,
            #        ((state['tree']['top'] - 100) - (state['monkey']['top'] - 28) )//boxes,

            #       state['tree']['dist'] // boxes, 
            #       state['monkey']['vel'] // vel_boxing,

            #       stats.mode(self.epoch_gravity)[0][0],
            #       action])
            
            arr = np.array([
                   (state['monkey']['top'] - 100) // boxes, 
                   (state['tree']['top'] - 28) // boxes,
                   state['tree']['dist'] // boxes, 
                   state['monkey']['vel'] // vel_boxing,

                   stats.mode(self.epoch_gravity)[0][0],
                   action])
            
            return arr
        return None

    def get_Q_score(self, state, action, prev_state):
        if state is None or prev_state is None or action is None or not self.fitted:
            return -1
        
        return self.rf.predict(np.array(self.state_action_to_array(state, action, prev_state)).reshape(1, -1))
    
    def set_Q_score(self, state, action, q, prev_state):
        arr = self.state_action_to_array(state, action, prev_state, set_epoch_graivty=True)
        if arr is not None:
            self.epoch_X = np.vstack((self.epoch_X, arr))
            self.y = np.append(self.y, q)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        prev_Q = self.get_Q_score(self.last_state, self.last_action, self.penultimate_state)
        
        swing_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 0, self.last_state) - prev_Q)
        
        jump_Q = prev_Q + self.learning_rate * (self.last_reward + self.discount_factor * self.get_Q_score(state, 1, self.last_state) - prev_Q)

        print "SWING %f, JUMP %f" % (swing_Q, jump_Q)
        
        # Pick the better Q score from the possible actions at s_{t+1}
        action_Qs = [swing_Q, jump_Q]
        best_action = 0
        if jump_Q > swing_Q:
            best_action = 1
            
        if self.train or jump_Q == swing_Q or npr.rand() < self.epsilon:
            new_action = (npr.rand() < 0.1) * 1
        else:
            new_action = best_action
        
        best_Q = action_Qs[new_action]
        
        if isinstance(best_Q, list):
            best_Q = best_Q[0]
        
        # Update the Q score for the last state and action
        self.set_Q_score(self.last_state, self.last_action, best_Q, self.penultimate_state)

        self.penultimate_state = self.last_state
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