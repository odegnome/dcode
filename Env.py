from dm_control import mujoco
import numpy as np
import tensorflow as tf

OBS_SPACE = 24
ACTION_SPACE = 4

class QuadEnv:
    def __init__(self, xml_path):
        self.physics = mujoco.Physics.from_xml_path(xml_path)
    
    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        done = True if self.physics.data.ncon > 0 else False
        reward = 1 if not done else -1000
        state = self.get_observation()
        return state, reward, done, []

    def get_observation(self):
        # Observation
        sensor_data = self.physics.data.sensordata
        orientation = self.physics.data.xquat[6]
        # velocities = self.physics.data.qvel
        state = np.concatenate((sensor_data, orientation))#, velocities))
        return state
    
    def reset(self):
        controls = [
            [0., 0., 0., 0., 0., 0.],
            [0.0, 0.0, 0.0, 0.0, 2., 3.],
            [0.36, 0.36, 0.33, 0.33, 2., 3.],
            [0.33, 0.33, 0.36, 0.36, -2., 3.]
        ]
        index = np.random.randint(0,4)
        with self.physics.reset_context():
            self.physics.set_control(controls[index][0:4])
            self.physics.data.qvel[1] = controls[index][4]
            self.physics.data.qvel[2] = controls[index][5]

        return self.get_observation()
    
    def render(self):
        return self.physics.render(height=720, width=1024, camera_id='fixed_camera')


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
        x = np.round(x, decimals=3)
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

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
    output = tf.keras.layers.Dense(4)(out)

    model = tf.keras.Model(actor_input, output)
    return model

def get_critic():
    critic_input_1 = tf.keras.layers.Input(shape=(OBS_SPACE,))
    critic_input_2 = tf.keras.layers.Input(shape=(ACTION_SPACE,))
    critic_input = tf.keras.layers.Concatenate()([critic_input_1, critic_input_2])
    out = tf.keras.layers.Dense(64,activation='tanh')(critic_input)
    out = tf.keras.layers.Dense(64,activation='tanh')(out)
    output = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model([critic_input_1, critic_input_2], output)
    return model
