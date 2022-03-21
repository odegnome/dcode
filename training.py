from dm_control import mujoco
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel('WARNING')

# Apply logging config.
logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Gives the shape of the observation space
OBS_SPACE = 34
# Gives the shape of the action space
ACTION_SPACE = 4
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

# Code for actor-critic networks

# Code for loading physics model of quad.xml
class QuadEnv:
    def __init__(self, xml_path):
        self.physics = mujoco.Physics.from_xml_path(xml_path)
    
    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        done = True if self.physics.data.ncon > 0 else False
        reward = 1 if done else -1000

        # Observation
        sensor_data = self.physics.data.sensordata
        orientation = self.physics.data.xquat[6]
        velocities = self.physics.data.qvel
        actuators = self.physics.data.actuator_force
        state = np.concatenate((sensor_data, orientation, velocities, actuators))
        return state, reward, done, []
    
    def reset(self):
        self.physics.reset()
        # Observation
        sensor_data = self.physics.data.sensordata
        orientation = self.physics.data.xquat[6]
        velocities = self.physics.data.qvel
        actuators = self.physics.data.actuator_force
        state = np.concatenate((sensor_data, orientation, velocities, actuators))
        return state


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=5000, batch_size=128):
        # Memory for Experience Replay
        self.buffer_capacity = buffer_capacity
        # Batch Size for training
        self.batch_size = batch_size
        # Gives the number of times .record() was called
        self.buffer_counter = 0

        # Set buffers for each attribute rather than using tuple
        self.state_buffer = np.zeros((buffer_capacity, OBS_SPACE))
        self.action_buffer = np.zeros((buffer_capacity, ACTION_SPACE))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, OBS_SPACE))
    
    def record(self, obs):
        """
        Takes (s,a,r,s') as input and assigns each value to their indivisual
        buffer.
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs[0]
        self.action_buffer[index] = obs[1]
        self.reward_buffer[index] = obs[2]
        self.next_state_buffer[index] = obs[3]

        self.buffer_counter += 1
    
    @tf.function
    def update(self, sbatch, abatch, rbatch, nsbatch):
        """
        sbatch: state_batch
        abatch: action_batch
        rbatch: reward_batch
        nsbatch: next_state_batch
        """
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = tf.cast(target_actor(nsbatch, training=True), dtype=tf.float64)
            y = rbatch + gamma * target_critic(
                tf.concat([nsbatch, target_actions],axis=1), training=True
            )
            critic_value = critic_model(tf.concat((sbatch,abatch),axis=1), training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = tf.cast(actor_model(sbatch, training=True), dtype=tf.float64)
            critic_value = critic_model(tf.concat([sbatch, actions],axis=1), training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

def get_actor():
    actor_input = tf.keras.layers.Input(shape=(OBS_SPACE,))
    out = tf.keras.layers.Dense(64,activation='tanh')(actor_input)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    output = tf.keras.layers.Dense(4, kernel_initializer='random_uniform')(out)

    model = tf.keras.Model(actor_input, output)
    return model

def get_critic():
    critic_input_1 = tf.keras.layers.Input(shape=(OBS_SPACE,))
    critic_input_2 = tf.keras.layers.Input(shape=(ACTION_SPACE,))
    critic_input = tf.keras.layers.concatenate([critic_input_1, critic_input_2])
    out = tf.keras.layers.Dense(64,activation='tanh')(critic_input)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    output = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model(critic_input, output)
    return model

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, LOWER_BOUND, UPPER_BOUND)

    return np.squeeze(legal_action)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# Code for training
env = QuadEnv('quad.xml')
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model= get_actor()
critic_model = get_critic()

target_actor= get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

MAX_EPISODES = 10
MAX_EPISODE_LEN = 12000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(5000, 64)

# Running the experiment

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

for episode in range(MAX_EPISODES+1):

    prev_state = env.reset()
    episodic_reward = 0

    for _ in range(MAX_EPISODE_LEN):
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    # if episode%100 == 0:
    #     actor_model.save(f'training/actor{episode}.tf', save_format='tf')
    #     critic_model.save(f'training/critic{episode}.tf', save_format='tf')
    #     target_actor.save(f'training/tactor{episode}.tf', save_format='tf')
    #     target_critic.save(f'training/tcritic{episode}.tf', save_format='tf')

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {} and Episode Reward ==> {}".format(episode, avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.savefig('rewards.png')
