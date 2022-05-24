import os

subdir = '24-05'
if not os.access(f'../training/{subdir}/', os.F_OK):
    os.mkdir(f'../training/{subdir}/')
if not os.access(f'../images/{subdir}/', os.F_OK):
    os.mkdir(f'../images/{subdir}/')
if not os.access(f'../logs/{subdir}/', os.F_OK):
    os.mkdir(f'../logs/{subdir}/')

import logging
# Apply logging config.
logging.basicConfig(
    filename=f'../logs/{subdir}/training.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Env import QuadEnv, Buffer, get_actor, get_critic
from sys import exit
import pickle

SEED = 103473

np.random.seed(SEED)
# Gives the shape of the observation space
OBS_SPACE = 29
# Gives the shape of the action space
ACTION_SPACE = 4
# Lower and upper bound of quadrotor actuators
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

def policy(state, noise):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()
    sampled_actions += noise

    # make sure action is within bounds
    legal_action = np.clip(sampled_actions, LOWER_BOUND, UPPER_BOUND)

    return np.squeeze(legal_action)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# def main(*args, **kwargs):
# Code for training
env = QuadEnv('quad.xml')
rng = np.random.default_rng(SEED)

actor_model = tf.keras.models.load_model(f'../training/03-05/actor10000')
critic_model = tf.keras.models.load_model(f'../training/03-05/critic10000')

target_actor = tf.keras.models.load_model(f'../training/03-05/tactor10000')
target_critic = tf.keras.models.load_model(f'../training/03-05/tcritic10000')

# actor_model = get_actor()
# critic_model = get_critic()

# target_actor= get_actor()
# target_critic = get_critic()

# Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# critic_optimizer_weights = np.load(f'../training/{subdir}/coptimizer.npy', allow_pickle=True)
# actor_optimizer_weights = np.load(f'../training/{subdir}/aoptimizer.npy', allow_pickle=True)

# Apply saved optimizer weights to critic_optimizer
# grad_vars = critic_model.trainable_variables
# zero_grads = [tf.zeros_like(w) for w in grad_vars]
# critic_optimizer.apply_gradients(zip(zero_grads, grad_vars))
# critic_optimizer.set_weights(critic_optimizer_weights)
# del grad_vars

# # Apply saved optimizer weights to actor_optimizer
# grad_vars = actor_model.trainable_variables
# zero_grads = [tf.zeros_like(w) for w in grad_vars]
# actor_optimizer.apply_gradients(zip(zero_grads, grad_vars))
# actor_optimizer.set_weights(actor_optimizer_weights)
# del grad_vars, zero_grads

rng = np.random.default_rng(seed=SEED)

EPISODE_START = 10001
MAX_EPISODES = 20000
MAX_EPISODE_LEN = 2000
BATCH_SIZE = 128
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
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

# Running the experiment
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
debug_info = {'loss':[], 'eulers':[], 'actions':[]}
for episode in range(EPISODE_START, MAX_EPISODES+1):
    prev_state = env.reset()
    episodic_reward = 0
    for _ in range(MAX_EPISODE_LEN):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        # action according to policy + noise
        noise = rng.normal(0.0, 0.1, size=4)
        action = policy(tf_prev_state, noise)
        # Recieve state and reward from environment.
        state, reward, done = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        critic_loss, actor_loss = buffer.learn()
        # logging.info(f'Euler Angles: {state[-3:]}')
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state

        debug_info['loss'].append((critic_loss.numpy().copy(), actor_loss.numpy().copy()))
        debug_info['eulers'].append(state[-3:].copy())
        debug_info['actions'].append(action.copy())
        # End this episode when `done` is True
        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 100 episodes
    avg_reward = np.mean(ep_reward_list[-100:])
    print("Episode * {} * Avg Reward is ==> {} and Episode Reward ==> {}".format(episode, avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)

    if episode%1000 == 0:
        actor_model.save(f'../training/{subdir}/actor{episode}.h5', save_format='h5')
        critic_model.save(f'../training/{subdir}/critic{episode}.h5', save_format='h5')
        target_actor.save(f'../training/{subdir}/tactor{episode}.h5', save_format='h5')
        target_critic.save(f'../training/{subdir}/tcritic{episode}.h5', save_format='h5')
        plt.plot(avg_reward_list, 'g-')
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.savefig(f'../images/{subdir}/rewards{episode}.png', transparent=True)
        # del ep_reward_list[:900]
        with open(f'../logs/{subdir}/info{episode}.pickle', 'w+b') as fp:
            pickle.dump(debug_info, fp)
        del debug_info
        debug_info = {'loss':[], 'eulers':[], 'actions':[]}
        np.save(f'../training/{subdir}/coptimizer{episode}.npy', np.asanyarray(critic_optimizer.get_weights(), dtype=object))
        np.save(f'../training/{subdir}/aoptimizer{episode}.npy', np.asanyarray(actor_optimizer.get_weights(), dtype=object))
