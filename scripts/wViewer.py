# I must say, this is working as expected, on the very first try!
from dm_control import composer, viewer
from QuadEnv import QuadEnv, FlyWithoutCollision
import numpy as np


rng = np.random.default_rng(478203)
def random_action():
    action = rng.random((4,))
    return action

if __name__  == "__main__":
    quad = QuadEnv()
    task = FlyWithoutCollision(quad)
    env = composer.Environment(task)

    viewer.launch(env, policy=random_action()) 