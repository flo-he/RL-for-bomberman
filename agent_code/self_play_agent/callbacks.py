import sys
import os 
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import deque

# import own modules
from .assembly import DuelingDQN
from .preprocessing import preprocess

# path to the training data, model, info
DQN_PATH = os.path.join(".", "data/models/duel_ddqn1.ckpt")
INFO_PATH = os.path.join(".", "data/train_info/info1.npy")

# GENERAL
N_FRAMES = 4
ACTIONS = {
    0: "RIGHT",
    1: "LEFT",
    2: "UP",
    3: "DOWN",
    4: "BOMB",
    5: "WAIT"
}
ACTION_SPACE_SIZE = len(ACTIONS)
INPUT_SHAPE = (15, 15, N_FRAMES)

# MODEL
DISCOUNT_RATE = 0.99
LEARNING_RATE = 0.001
MOMENTUM = 0.90



def setup(self):
    '''
    Setup the agent: Specify its inputs and possible actions, build
    the network or load the network with already trained weights.
    '''
    try:
        # for states
        self.recent_frames = deque(maxlen=N_FRAMES)
        self.stacked_frames = np.empty(shape=(15, 15, N_FRAMES), dtype=np.int8)
          
        # HYPERPARAMETERS
        self.learning_rate = LEARNING_RATE
        self.momentum = MOMENTUM
        
        # give the agent the actions und the input shape for the dqn
        self.actions = ACTIONS
        self.action_space_size = ACTION_SPACE_SIZE
        self.input_shape = INPUT_SHAPE

        # assemble the networks
        self.online_dqn = DuelingDQN("online_DuelingDQN", self.action_space_size, self.input_shape, self.learning_rate, self.momentum)
        #self.target_dqn = DuelingDQN("target_DuelingDQN", self.action_space_size, self.input_shape, self.learning_rate, self.momentum)
        self.saver = tf.train.Saver()

        # start a session. add config=t_config if train multiple instances on one GPU 
        self.sess = tf.Session()
        print("Tensorflow Session started.")
            
        # get model state info
        info = np.load(INFO_PATH)
        self.episode = info[0]
        #self.epsilon = info[1]
        self.epsilon = 0.05
        #print("Episode, Epsilon, Decay, min_Epsilon = ", info)

        # restore tensorflow session
        try:
            self.saver.restore(self.sess, DQN_PATH) 
            print("Session restored.")
        except Exception as e:
            raise RuntimeError(f"{e}")
        
             
    except Exception as e:
        raise RuntimeError(f"{e}")

def act(self):
    try:  
        # if game round starts, beginning history is 4x beginning state frame
        preprocessed_img = preprocess(self.game_state)
        if self.game_state['step'] == 1:
            for _ in range(N_FRAMES):
                self.recent_frames.append(preprocessed_img)
        else:
            self.recent_frames.append(preprocessed_img)

        # the state (neural network input) are 4 frames stacked along the third dimension
        # to give the agent some sense of time
        # (converting deque to a list beforehand increases performance)
        for i in range(N_FRAMES):
            self.stacked_frames[:, :, i] = self.recent_frames[i]

        # EPSILON GREEDY
        # get current epsilon

        # sample a random number from 0 to 1. If it's smaller than epsilon,
        # execute a random action, else let the DQN decide on the next action to take
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.choice([*self.actions], p=[.22, .22, .22, .22, .08, .04])
            self.next_action = self.actions[action_idx]
            #self.next_action = self.game_state['user_input']
            # print(f"Action chosen at random: {self.actions[action_idx]}")
        else:
            q_values = self.sess.run(self.online_dqn.output_layer, feed_dict={self.online_dqn.input_layer: [self.stacked_frames]})
            #print(q_values)
            #q_dict = {act: val for act, val in zip(self.actions.items(), *q_values)}
            #print(q_dict)
            action_idx = np.argmax(q_values)
            self.next_action = self.actions[action_idx]
            #print(f"Action chosen by agent: {self.actions[action_idx]}")

    except Exception as e:
        raise RuntimeError(f"{e}")

def reward_update(self):
    pass

def end_of_episode(self):
    pass


