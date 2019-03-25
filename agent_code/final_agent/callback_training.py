import numpy as np
import tensorflow as tf
import sys
import os 
import pickle
from collections import deque
import time

import threading
import queue

# uncomment if GPU available and one wants to train several instances on one GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#t_config = tf.ConfigProto()
#t_config.gpu_options.allow_growth = True

# import own modules
from .assembly import DuelingDQN, ExperienceBuffer
from .preprocessing import preprocess, get_reward 

# path to the training data, model, info
BUFFER_PATH = os.path.join(".", "data/experience/replay_memory0.file")
DQN_PATH = os.path.join(".", "data/models/duel_ddqn0.ckpt")
INFO_PATH = os.path.join(".", "data/train_info/info0.npy")
TB_DIR = os.path.join(".", "tb/Duel_DDQN/1E-3_32_1E-5_2")


# GENERAL
TRAINING = True
USE_SAVED_MODEL = True
SKIP_EPISODES = 10000
BATCH_SIZE = 32
N_MEM = 2500000
N_FRAMES = 4
# epsilon is a tuple of starting epsilon value, decay, min epsilon
EPSILON = np.array([1.0, 0.000001, 0.1])
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

TAU = 0.00001
SAVE_STEP = 5000
INFO_STEP = 100

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
        self.logger.debug('Successfully entered setup code')

        # training flag
        self.TRAINING = TRAINING
        # if there's an already partially learned network and Experience buffer, resume training
        self.USE_SAVED_MODEL = USE_SAVED_MODEL
        
        # frame stack
        self.recent_frames = deque(maxlen=N_FRAMES)
        self.stacked_frames = np.empty(shape=(15, 15, N_FRAMES), dtype=np.int8)

        # init some parameters
        if self.TRAINING:
            self.skip_episodes = SKIP_EPISODES
            self.batch_size = BATCH_SIZE
            self.discount_rate = DISCOUNT_RATE
            # for rewarding and metrics
            self.max_Q = 0
            self.mean_max_Qs = []
            self.time_steps = []
            self.score = []
            self.losses = []
            self.rewards = []
            self.save_step = SAVE_STEP
            self.info_step = INFO_STEP
        else:
            # use cpu if not training
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # HYPERPARAMETERS
        self.learning_rate = LEARNING_RATE
        self.momentum = MOMENTUM
        self.tau = TAU
        
        # give the agent the actions und the input shape for the dqn
        self.actions = ACTIONS
        self.action_space_size = ACTION_SPACE_SIZE
        self.input_shape = INPUT_SHAPE

        # assemble the networks
        self.online_dqn = DuelingDQN("online_DuelingDQN", self.action_space_size, self.input_shape, self.learning_rate, self.momentum)
        self.target_dqn = DuelingDQN("target_DuelingDQN", self.action_space_size, self.input_shape, self.learning_rate, self.momentum)
        self.saver = tf.train.Saver()

        # start a session. add config=t_config if train multiple instances on one GPU 
        self.sess = tf.Session()
        print("Tensorflow Session started.")
            
        if self.TRAINING:
            # keep track of some metrics for tensorboard (t_x for tensor x)
            self.t_loss = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="loss_for_metrics")
            self.t_reward = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="reward_for_metrics")
            self.t_timestep = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="timestep_for_metrics")
            self.t_score = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="score_for_metrics")
            self.t_qval = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32, name="qval_for_metrics")

            self.loss_summ = tf.summary.scalar("Loss", self.t_loss)
            self.rew_summ = tf.summary.scalar("Reward", self.t_reward)
            self.ts_summ = tf.summary.scalar("Time Step", self.t_timestep)
            self.score_summ = tf.summary.scalar("Score", self.t_score)
            self.q_summ = tf.summary.scalar("max Q Value", self.t_qval)

            self.merged_summ = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(TB_DIR)

            try:
                self.sess.run(tf.global_variables_initializer())

                # copy operations for the tensorflow session, to copy online to target DQN
                # (assign all trainable weights of online DQN to target DQN)
                # update target weights every episode by a small amount (tau hyperparameter)
                with tf.variable_scope("Target_Update"):
                    self.copy_ops = [
                        target_var.assign(self.tau*self.online_dqn.trainable_vars_by_name[var_name].value() + (1.-self.tau)*target_var.value()) 
                            for var_name, target_var in self.target_dqn.trainable_vars_by_name.items()
                        ]
                    self.copy_online_to_target = tf.group(*self.copy_ops, name="Copy_Op")
            except Exception as e:
                raise RuntimeError(e)

            # initialize a writer to gather info for tensorboard
            try:
                self.writer.add_graph(self.sess.graph)
            except Exception as e:
                raise RuntimeError(e)
            
            if self.USE_SAVED_MODEL:
                if self.TRAINING:
                    try:
                        # load an already existing buffer from the hard drive
                        with open(BUFFER_PATH, "rb") as f:
                            self.Buffer = pickle.load(f)
                    except Exception as e:
                        print("Failed to load buffer from file.", e)

                # get model state info
                info = np.load(INFO_PATH)
                self.episode = info[0]
                self.epsilon = info[1:]
                #self.epsilon = [0.01, 0.000001, 0.01]
                print("Episode, Epsilon, Decay, min_Epsilon = ", info)

                # restore tensorflow session
                self.saver.restore(self.sess, DQN_PATH) 
                print("Session restored")
            else:
                try:
                    # init experience buffer
                    self.Buffer = ExperienceBuffer(n_memories=N_MEM, state_shape=INPUT_SHAPE)
                    print("Buffer_created")
                except Exception as e:
                    print(e)

                # init Epsilon
                self.epsilon = EPSILON
                self.episode = -SKIP_EPISODES

                # init the Networks
                try:
                    self.sess.run(self.online_dqn.init)
                except Exception as e:
                    print(e)
                print("Initialized online_DQN")
                try:
                    self.sess.run(self.target_dqn.init)
                except Exception as e:
                    print(e)
                print("Initialized target_DQN") 

            # create a lock for the buffer
            self.SharedLock = threading.RLock()
            # mini batch queue
            self.batch_queue = queue.Queue(maxsize=100)
            # batch thread 
            self.BatchThread = threading.Thread(target=self.Buffer.create_mini_batch, args=(self.batch_size, self), daemon=True)

            self.time = time.clock()
        self.logger.debug('Successfully exited setup code')
    except Exception as e:
        raise RuntimeError(f"{e}")

def act(self):
    try:
        # for rewarding
        self.last_pos = self.game_state["self"][:2]
        
        if self.TRAINING and (self.episode >= 0) and not self.BatchThread.is_alive():
            self.BatchThread.start()
            
        # if game round starts, beginning history is 4x beginning state frame
        preprocessed_img = preprocess(self.game_state)
        if self.game_state['step'] == 1:
            for _ in range(N_FRAMES):
                self.recent_frames.append(preprocessed_img)
        elif not self.TRAINING:
            self.recent_frames.append(preprocessed_img)

        # the state (neural network input) are 4 frames stacked along the third dimension
        # to give the agent some sense of time
        # (converting deque to a list beforehand increases performance)
        for i in range(N_FRAMES):
            self.stacked_frames[:, :, i] = self.recent_frames[i]
        
        # EPSILON GREEDY
        # get current epsilon
        eps = np.max(self.epsilon)
        # update epsilon if training has begun
        if self.TRAINING and (self.epsilon[0] >= eps) and (eps > self.epsilon[2]):
            if (0 < self.episode) or self.USE_SAVED_MODEL:
                self.epsilon[0] = eps - self.epsilon[1]
        
        # sample a random number from 0 to 1. If it's smaller than epsilon,
        # execute a random action, else let the DQN decide on the next action to take
        if np.random.rand() <= eps:
            action_idx = np.random.choice([*self.actions], p=[.22, .22, .22, .22, .08, .04])
            self.next_action = self.actions[action_idx]
            #self.next_action = self.game_state['user_input']
            # print(f"Action chosen at random: {self.actions[action_idx]}")
        else:
            q_values = self.sess.run(self.online_dqn.output_layer, feed_dict={self.online_dqn.input_layer: [self.stacked_frames]})
            #q_dict = {act: val for act, val in zip(self.actions.items(), *q_values)}
            #print(q_dict)
            
            action_idx = np.argmax(q_values)
            self.max_Q += q_values[0][action_idx]
            self.next_action = self.actions[action_idx]
            #print(f"Action chosen by agent: {self.actions[action_idx]}")

        
        # begin to gather transition info
        if self.TRAINING:
            self.SharedLock.acquire()
            self.Buffer.states.push(self.stacked_frames)
            self.Buffer.actions.push(action_idx)
    except Exception as e:
        raise RuntimeError(f"{e}")

def reward_update(self):
    try:
        # get the next state
        next_frame = preprocess(self.game_state)
        self.recent_frames.append(next_frame)
        
        for i in range(N_FRAMES):
            self.stacked_frames[:, :, i] = self.recent_frames[i]
        #next_state = np.dstack(list(self.recent_frames))

        # get reward
        #print(f"---- TIME_STEP: {self.game_state['step']-1} ----")
        reward = get_reward(self)
        
        # push elements in buffer
        self.Buffer.rewards.push(reward)
        self.Buffer.next_states.push(self.stacked_frames)
        self.Buffer.ep_end.push(False)

        self.SharedLock.release()

        # for metrics
        self.rewards.append(reward)
        
        # perform training step
        if (self.episode > 0) or self.USE_SAVED_MODEL:
            self.online_dqn.train(self)

    except Exception as e:
        raise RuntimeError(f"{e}")

def end_of_episode(self):
    try:
        # update episode counter
        self.episode += 1

        next_frame = preprocess(self.game_state)
        self.recent_frames.append(next_frame)

        for i in range(N_FRAMES):
            self.stacked_frames[:, :, i] = self.recent_frames[i]

        # get reward
        reward = get_reward(self)

        # push elements in buffer
        self.Buffer.rewards.push(reward)
        self.Buffer.next_states.push(self.stacked_frames)
        self.Buffer.ep_end.push(True)

        # print buffer size if not max capacity
        if (self.episode % self.info_step == 0):
            if (self.Buffer.get_size() < self.Buffer.n_memories):
                print(f"Episode: {self.episode}, Buffer_size: {self.Buffer.size}")

        self.SharedLock.release()

        # perform training step
        if (self.episode > 0) or self.USE_SAVED_MODEL:
            self.online_dqn.train(self)

        # get training info/metrics 
        time_step = self.game_state["step"]
        mean_max_Q = self.max_Q / time_step
        self.mean_max_Qs.append(mean_max_Q)
        self.max_Q = 0

        self.score.append(self.game_state["self"][4])
        self.time_steps.append(time_step)
        self.rewards.append(reward)

        # start training the model after specified skip of episodes (to fill buffer)
        if (self.episode > 0) or self.USE_SAVED_MODEL:
            
            # print training info
            if self.episode % self.info_step == 0:
                # get training metrics
                avg_loss = np.mean(self.losses)
                avg_score = np.mean(self.score)
                avg_reward = np.mean(self.rewards)
                avg_time_steps = np.mean(self.time_steps)

                # empty the lists
                self.score.clear()
                self.losses.clear()
                self.rewards.clear()
                self.time_steps.clear()

                # fill tensors
                self.sess.run([
                        self.t_loss.assign(avg_loss),
                        self.t_score.assign(avg_score),
                        self.t_reward.assign(avg_reward),
                        self.t_timestep.assign(avg_time_steps)
                    ])

                if self.mean_max_Qs:
                    avg_mm_Q = np.mean(self.mean_max_Qs)
                    self.sess.run(self.t_qval.assign(avg_mm_Q))
                    self.mean_max_Qs.clear()
                
                summary = self.sess.run(self.merged_summ)
                self.writer.add_summary(summary, global_step=self.episode)
                
                t = time.clock()-self.time
                print(f"Episode: {self.episode} --AVERAGE-- Loss: {avg_loss:.4f} | reward: {avg_reward:.3f} | Q_value: {avg_mm_Q:.3f} | TS: {avg_time_steps:.0f} | Score: {avg_score:.2f} | Eps: {self.epsilon[0]:.2f} | took {t:.2f}s")
                
                # save progress
                if self.episode % self.save_step == 0:
                    self.saver.save(self.sess, DQN_PATH)
                    print("Saved Model")
                    try:
                        with self.SharedLock:
                            self.Buffer.save_to_disc(BUFFER_PATH)
                        print("Saved Buffer.")
                    except Exception as e:
                        raise RuntimeError(f"{e}")
                    try:
                        np.save(file=INFO_PATH, arr=np.concatenate([[self.episode], self.epsilon]))
                    except Exception as e:
                        raise RuntimeError(f"{e}")
                        
                self.time = time.clock()
    except Exception as e:
        raise RuntimeError(f"{e}")

                    
        



