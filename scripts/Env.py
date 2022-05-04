from dm_control import mujoco
import numpy as np
import tensorflow as tf
from dm_control.utils.transformations import quat_to_euler
from dm_control.mujoco.wrapper.mjbindings import mjlib

OBS_SPACE = 29
ACTION_SPACE = 4
SEED = 103473

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

class QuadEnv:
    def __init__(self, xml_path):
        self.physics = mujoco.Physics.from_xml_path(xml_path)
    
    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        done = True if self.physics.data.ncon > 0 else False
        reward = self.get_reward()
        state = self.get_observation()
        return state, reward, done, []
    
    def get_reward(self):
        # Penalise for orientation greater than pi/6
        atitude = self.physics.data.xquat[6].copy()
        euler = quat_to_euler(atitude)
        norm = np.sqrt(np.sum(np.square(euler[:2])))
        reward = -np.round(norm, 3)

        # Penalise if distance is less than 1
        sensor_array = self.physics.data.sensordata[6:].copy()
        sensor_array[:] = sensor_array/3
        for index, x in enumerate(sensor_array):
            if x<0.34:
                sensor_array[index] = -14
        reward += np.sum(sensor_array)

        # Penalty for collision, also ends the episode
        if self.physics.data.ncon > 0:
            reward -= 10000
        return reward

    def get_observation(self):
        # Observation
        sensor_data = self.physics.data.sensordata.copy()
        sensor_data[6:] = sensor_data[6:]/3
        orientation = np.zeros(9)
        mjlib.mju_quat2Mat(orientation, self.physics.data.xquat[6].copy())
        # velocities = self.physics.data.qvel
        state = np.concatenate((sensor_data, orientation))#, velocities))
        return state
    
    def reset(self):
        # controls = [
        #     [0., 0., 0., 0., 0., 0.],
        #     [0.0, 0.0, 0.0, 0.0, 2., 3.],
        #     [0.36, 0.36, 0.33, 0.33, 2., 3.],
        #     [0.33, 0.33, 0.36, 0.36, -2., 3.]
        # ]
        # index = rng.integers(0,3, endpoint=True)
        # with self.physics.reset_context():
        #     self.physics.set_control(controls[index][0:4])
        #     self.physics.data.qvel[1] = controls[index][4]
        #     self.physics.data.qvel[2] = controls[index][5]
        
        with self.physics.reset_context():
            self.physics.data.qvel[:] = rng.normal(0.0, 0.1, size=6)

        return self.get_observation()
    
    def render(self):
        return self.physics.render(height=720, width=1024, camera_id='fixed_camera')


class Buffer:
    def __init__(self, models, optimizers, gamma=0.99, OBS_SPACE=24,
                ACTION_SPACE=4, buffer_capacity=5000, batch_size=128):
        # Memory for Experience Replay
        self.buffer_capacity = buffer_capacity
        # Batch Size for training
        self.batch_size = batch_size
        # Gives the number of times .record() was called
        self.buffer_counter = 0
        # Hyperparameters
        self.gamma = gamma
        self.OBS_SPACE = OBS_SPACE
        self.ACTION_SPACE = ACTION_SPACE

        # Set buffers for each attribute rather than using tuple
        self.state_buffer = np.zeros((self.buffer_capacity, self.OBS_SPACE))
        self.action_buffer = np.zeros((self.buffer_capacity, self.ACTION_SPACE))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.OBS_SPACE))

        self.actor_model = models[0]
        self.critic_model = models[1]
        self.target_actor = models[2]
        self.target_critic = models[3]

        self.actor_optimizer = optimizers[0]
        self.critic_optimizer = optimizers[1]
    
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
            target_actions = tf.cast(self.target_actor(nsbatch, training=True), dtype=tf.float64)
            y = rbatch + self.gamma * self.target_critic(
                [nsbatch, target_actions], training=True
            )
            critic_value = self.critic_model((sbatch,abatch), training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = tf.cast(self.actor_model(sbatch, training=True), dtype=tf.float64)
            critic_value = self.critic_model([sbatch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        return critic_loss, actor_loss
        
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

        losses = self.update(state_batch, action_batch, reward_batch, next_state_batch)
        return losses

def get_actor():
    actor_input = tf.keras.layers.Input(shape=(OBS_SPACE,))
    out = tf.keras.layers.BatchNormalization()(actor_input)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    output = tf.keras.layers.Dense(4)(out)

    model = tf.keras.Model(actor_input, output)
    return model

def get_critic():
    critic_input_1 = tf.keras.layers.Input(shape=(OBS_SPACE,))
    critic_input_2 = tf.keras.layers.Input(shape=(ACTION_SPACE,))
    critic_input = tf.keras.layers.Concatenate()([critic_input_1, critic_input_2])
    out = tf.keras.layers.BatchNormalization()(critic_input)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    output = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model([critic_input_1, critic_input_2], output)
    return model
