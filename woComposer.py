from dm_control import mjcf
from tensorflow.keras import Sequential, layers

class QuadEnv:
    def __init__(self):
        self.env = mjcf.from_path('quad.xml')
    
    def is_colliding(self):
        """Check for collision and return status. True means there is collision"""
        return
    
    def collate_obs(self):
        """Collect all the sensor values together to be passed as obs"""
        return

class DQNAgent:
    def __init__(self):
        pass

    def define_models(self):
        actor = Sequential()
        actor.add(layers.Dense(32,activation='tanh',input_shape=(20,)))
        actor.add(layers.Dense(32,activation='tanh'))
        actor.add(layers.Dense(4,activation='sigmoid'))

        critic = Sequential()
        critic.add(layers.Dense(32,activation='tanh',input_shape=(20,)))
        critic.add(layers.Dense(32,activation='tanh'))
        critic.add(layers.Dense(1))
        return actor,critic
