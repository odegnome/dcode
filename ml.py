from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import TensorBoard 

class Agent:
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
    
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
