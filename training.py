import logging
# Apply logging config.
logging.basicConfig(
    filename='logs/training.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Env import QuadEnv, OUActionNoise, Buffer, get_actor, get_critic

np.random.seed(103473)
# Gives the shape of the observation space
OBS_SPACE = 30
# Gives the shape of the action space
ACTION_SPACE = 4
# Lower and upper bound of quadrotor actuators
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

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
std_dev = 0.5
ou_noise = OUActionNoise(mean=np.zeros(4), std_deviation=float(std_dev) * np.ones(4))

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

MAX_EPISODES = 500
MAX_EPISODE_LEN = 12000
BATCH_SIZE = 64
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
epsilon = 1
buffer = Buffer(
        models=[actor_model, critic_model, target_actor, target_critic],
        optimizers=[actor_optimizer,critic_optimizer],
        gamma=gamma,
        OBS_SPACE=OBS_SPACE,
        ACTION_SPACE=ACTION_SPACE,
        buffer_capacity=10000,
        batch_size=BATCH_SIZE
    )

# Running the experiment
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
for episode in range(MAX_EPISODES+1):
    prev_state = env.reset()
    episodic_reward = 0
    for _ in range(MAX_EPISODE_LEN):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        # Take random action or take action according to policy
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

    if episode%100 == 0:
        actor_model.save(f'training/actor{episode}', save_format='tf')
        critic_model.save(f'training/critic{episode}', save_format='tf')
        target_actor.save(f'training/tactor{episode}', save_format='tf')
        target_critic.save(f'training/tcritic{episode}', save_format='tf')
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.savefig(f'images/rewards{episode}.png')

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {} and Episode Reward ==> {}".format(episode, avg_reward, episodic_reward))
    avg_reward_list.append(avg_reward)
