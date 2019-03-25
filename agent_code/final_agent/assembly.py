import numpy as np
import tensorflow as tf
import pickle

class DuelingDQN(object):
    '''
    Builds a Deep-Q-Network based on a CNN architecture for Deep-Q-Learning using tensorflow low-level-api.
    Uses Dueling Strategy, where the network predicts Value function V(s) and Action Advantage A(s,a) separately,
    then combines the prediction to the final Q(s,a) value.
    '''

    def __init__(self, name, action_size, input_shape, learning_rate, momentum):
        try:
            self.action_size = action_size
            self.input_shape = input_shape
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.layer_init = tf.variance_scaling_initializer()
            self.name = name


            with tf.variable_scope(name) as main_scope:
                with tf.variable_scope("Assembly"):
                    # Input layer: takes state(s) (image(s) with input_shape)
                    self.input_layer = tf.placeholder(dtype=np.int8, shape=(None, *input_shape), name="Stacked_Frames")
        
                    # First Convolution
                    self.conv1_layer = tf.layers.conv2d( 
                        inputs = self.input_layer/128,
                        filters = 16,
                        kernel_size = (4, 4),
                        strides = 2,
                        padding = 'same',
                        activation = tf.nn.relu,
                        kernel_initializer = self.layer_init,
                        name = "conv1"
                    ) 

                    # Second Convolution
                    self.conv2_layer = tf.layers.conv2d( 
                        inputs = self.conv1_layer,
                        filters = 32,
                        kernel_size = (2, 2),
                        strides = 1,
                        padding = 'same',
                        activation = tf.nn.relu,
                        kernel_initializer = self.layer_init,
                        name = "conv2"
                    ) 

                    # flatten the last convolutional layer to match the next hidden layers inputs
                    self.conv2_flattened = tf.reshape(self.conv2_layer, shape=(-1, tf.reduce_prod(self.conv2_layer.shape[1:])), name="flatten")

                    # Dueling DQN: Split into two streams V(s) and A(s,a)
                    with tf.variable_scope("Dueling_Architecture"):
                        # V(s) fully connected layer
                        self.value_func_layer = tf.layers.dense(
                            inputs = self.conv2_flattened,
                            units = 512,
                            activation = tf.nn.relu,
                            kernel_initializer = self.layer_init,
                            name = "V_fc"
                        )
     
                        # result of the V(s) layer, actually predicted V(s) value
                        self.value_func = tf.layers.dense(
                            inputs = self.value_func_layer,
                            units = 1,
                            activation = None,
                            kernel_initializer = self.layer_init,
                            name = "V_out"
                        )

                        # A(s, a) stream
                        self.adv_func_layer = tf.layers.dense(
                            inputs = self.conv2_flattened,
                            units = 512,
                            activation= tf.nn.relu,
                            kernel_initializer = self.layer_init,
                            name = "A_fc"
                        )

                        # predicted advantages
                        self.adv_func = tf.layers.dense(
                            inputs = self.adv_func_layer,
                            units = self.action_size,
                            activation = None,
                            kernel_initializer = self.layer_init,
                            name = "A_out"
                        )

   
                        # aggregation layer
                        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
                        self.output_layer = tf.add(self.value_func, tf.subtract(self.adv_func, tf.reduce_mean(self.adv_func, axis=1, keepdims=True)), name="Q_values")

    

                with tf.variable_scope("Train"):
                    # entry point for the "ground truth" Q_values
                    self.y_Q = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Target_Q")
                    self.actions_input = tf.placeholder(dtype=tf.int32, shape=[None], name="Actions_Input")
                    
                    # compute Q values prediction
                    self.one_hot = tf.one_hot(self.actions_input, self.action_size)
                    self.Q = tf.reduce_sum(self.output_layer * self.one_hot, axis=1, keepdims=True, name="Q_Values")
                    #self.Q_summary = tf.summary.tensor('Q_Values', self.Q)
                    
                    # use huber loss function for smoother learning
                    with tf.variable_scope("Loss"):
                        self.loss = tf.losses.huber_loss(self.y_Q, self.Q)
                        #self.loss_summary = tf.summary.scalar('huber_loss', self.loss)

                    # use MomentumOptimizer with nesterov initialization (advanced SGD)
                    with tf.variable_scope("Momentum_Optimier"):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
                        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
                        self.training_op = self.optimizer.minimize(self.loss)
                    
                    

                with tf.variable_scope("trainable_vars"):    
                    # get trainable variables of the network, to copy weights between
                    # online and target DQN
                    self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope.name)
                    self.trainable_vars_by_name = {var.name[len(main_scope.name):]: var for var in self.trainable_vars}

                with tf.variable_scope("Global_Initializer"):
                    self.init = tf.global_variables_initializer()
        except:
            raise RuntimeError(f"Assembly of {name} failed.")

        print(f"Assembled the {name}")
    
    @staticmethod
    def train(agent):
        '''
        Performs one training step, i.e. feeds the DQN one batch from the experience buffer and trains it. Uses
        Double DQN algorithm (DDQN) to reduce overestimation of Q-Values.
        '''   
        # get the batch
        states, actions, rewards, next_states, continues = agent.batch_queue.get()

        # compute q values for the next states
        next_state_q, target_q = agent.sess.run(
                                    [agent.online_dqn.output_layer, agent.target_dqn.output_layer],
                                    feed_dict={
                                        agent.online_dqn.input_layer: next_states,
                                        agent.target_dqn.input_layer: next_states
                                    }
                                )

        # online dqn chooses best action for the next state
        best_action = np.argmax(next_state_q, axis=1)

        # compute the "ground truth" q value: reward + : if the episode ended the future reward is 0 (use of continues)
        # else it's the discount rate times the target q value of the next action chosen by the online dqn (1-step-Q-Learning update DDQN)
        dim_0_indices = np.arange(target_q.shape[0])
        y_val = rewards.reshape(-1, 1) + continues.reshape(-1, 1) * agent.discount_rate * target_q[dim_0_indices, best_action].reshape(-1, 1)

        # actual training step
        loss, _ , __= agent.sess.run(
                    [agent.online_dqn.loss, agent.online_dqn.training_op, agent.copy_online_to_target],
                    feed_dict = {
                        agent.online_dqn.input_layer: states,
                        agent.online_dqn.actions_input: actions,
                        agent.online_dqn.y_Q: y_val
                    }
                )
        agent.losses.append(loss)

class MemArray(object):
    '''Base container for the different elements of the ExperienceBuffer. Acts like a ringbuffer.'''

    def __init__(self, maxlen, dtype, shape=None):
        # initialize array with given shape or 1d
        if shape is not None:
            self.array = np.empty(shape=(maxlen, *shape), dtype=dtype)
        else:
            self.array = np.empty(shape=maxlen, dtype=dtype)

        self.maxlen = maxlen
        self.size = 0
        self.idx = 0
        
    
    def push(self, data):
        # add element to the deque
        self.array[self.idx] = data
        # increment size if size < maxlen
        self.size = min(self.size + 1, self.maxlen)
        # increment idx, if size == maxlen, start from the beginning of the container
        self.idx = (self.idx + 1) % self.maxlen
        
class ExperienceBuffer(object):
    '''
    Create an experience buffer with max number of memories 'n_memories'
    consisting of five MemArrays, which will hold instances of 
    state, action, reward, next_state, episode_end, where the states are (stacks of) the 
    preprocessed "images" of the game grid. This buffer gets filled with each
    time step, episode after episode and will be used to train the DQN networks.
    '''

    
    def __init__(self, n_memories, state_shape):
        # initialize with max number of memories, empty container and size.
        self.n_memories = n_memories
        self.states = MemArray(maxlen=n_memories, dtype=np.int8, shape=state_shape)
        self.actions = MemArray(maxlen=n_memories, dtype=np.int8)
        self.rewards = MemArray(maxlen=n_memories, dtype=np.float32)
        self.next_states = MemArray(maxlen=n_memories, dtype=np.int8, shape=state_shape)
        self.ep_end =  MemArray(maxlen=n_memories, dtype=np.bool)
        self.size = 0
        self.full = False
    

    def get_size(self):
        # size gets handled in the MemArrays, has to be checked and updated from time to time
        if not self.full:
            sizes = np.array([arr.size for arr in [self.states, self.actions, self.rewards, self.next_states, self.ep_end]])
            assert np.lib.arraysetops.unique(sizes).size == 1, "Container in ExperienceBuffer instance are not of same size."

            #assert np.unique(sizes).size == 1, "Container in ExperienceBuffer instance are not of same size." <- np.unique deprecated
        
            # update own size
            size = sizes[0]
            self.size = size
        else:
            return self.n_memories

        if self.size == self.n_memories:
            self.full = True

        return size
        
    def create_mini_batch(self, batch_size, agent):
        # sample from the experience of batch_size length with replacement (faster than using np.random.choice with or without replacement)
        # update size
        while(1):
            with agent.SharedLock:
                #print("create mini batch acquired lock")
                if not self.full:
                    size = self.get_size()
                # sample indices up to own size
                indices = np.random.randint(self.size, size=batch_size)

                # get the mini batch
                states = self.states.array[indices]
                actions = self.actions.array[indices]
                rewards = self.rewards.array[indices]
                next_states = self.next_states.array[indices]
                ep_end = self.ep_end.array[indices]
                # 1*np.logical_not(ep_end) converts ep_end from boolean to int and swaps true with false
                # (episode ended -> episode continues). Simply more convenient for further computations
            agent.batch_queue.put([states, actions, rewards, next_states, 1*np.logical_not(ep_end)])
            #print("create mini batch released lock")


    def save_to_disc(self, path):
        # save the buffer to the hard drive as a binary file
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Saving the buffer failed: ", e)
        

        

    
    
    
