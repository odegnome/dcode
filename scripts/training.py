import logging
# Apply logging config.
logging.basicConfig(
    filename='../logs/training.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Env import QuadEnv, OUActionNoise, Buffer, get_actor, get_critic
from sys import exit
import pickle
import os

SEED = 103473

np.random.seed(SEED)
# Gives the shape of the observation space
OBS_SPACE = 23
# Gives the shape of the action space
ACTION_SPACE = 4
# Lower and upper bound of quadrotor actuators
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

def policy(state, noise):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, LOWER_BOUND, UPPER_BOUND)

    return np.squeeze(legal_action)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# def main(*args, **kwargs):
# Code for training
env = QuadEnv('quad.xml')
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(4), std_deviation=float(std_dev) * np.ones(4))

# actor_model = tf.keras.models.load_model('../training/new/actor2000')
# critic_model = tf.keras.models.load_model('../training/new/critic2000')

# target_actor = tf.keras.models.load_model('../training/new/tactor2000')
# target_critic = tf.keras.models.load_model('../training/new/tcritic2000')

actor_model = get_actor()
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

rng = np.random.default_rng(seed=SEED)

MAX_EPISODES = 1000
MAX_EPISODE_LEN = 12000
BATCH_SIZE = 64
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
# Exploration eploitation tradeoff
epsilon = 1
# decay_rate = 0.999
# Buffer to hold past observation
buffer = Buffer(
        models=[actor_model, critic_model, target_actor, target_critic],
        optimizers=[actor_optimizer,critic_optimizer],
        gamma=gamma,
        OBS_SPACE=OBS_SPACE,
        ACTION_SPACE=ACTION_SPACE,
        buffer_capacity=50000,
        batch_size=BATCH_SIZE
    )

# logging results subdirectory
subdir = '30-04'
if not os.access(f'../training/{subdir}/', os.F_OK):
    os.mkdir(f'../training/{subdir}/')
if not os.access(f'../images/{subdir}/', os.F_OK):
    os.mkdir(f'../images/{subdir}/')
if not os.access(f'../logs/{subdir}/', os.F_OK):
    os.mkdir(f'../logs/{subdir}/')
# Running the experiment
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
# Lin acc and Ang vel data
debug_info = {'loss':[], 'info':[]} 
for episode in range(MAX_EPISODES+1):
    prev_state = env.reset()
    episodic_reward = 0
    for timestep in range(MAX_EPISODE_LEN):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        # action according to policy + noise
        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        current_loss = buffer.learn()
        logging.info(f'Euler Angles: {state[-3:]}')
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state
        # End this episode when `done` is True
        if done:
            break

    if episode%200 == 0:
        actor_model.save(f'../training/{subdir}/actor{episode}', save_format='h5')
        critic_model.save(f'../training/{subdir}/critic{episode}', save_format='h5')
        target_actor.save(f'../training/{subdir}/tactor{episode}', save_format='h5')
        target_critic.save(f'../training/{subdir}/tcritic{episode}', save_format='h5')
        plt.plot(avg_reward_list, 'g-')
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.savefig(f'../images/{subdir}/rewards{episode}.png')
        del ep_reward_list[:150]

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-50:])
    print("Episode * {} * Avg Reward is ==> {} and Episode Reward ==> {}".format(episode, avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)

with open('../logs/{subdir}/info.pickle', 'wb') as fp:
    pickle.dump(debug_info, fp)
