from random import random
import tensorflow as tf
from Env import QuadEnv, get_actor
import logging
import numpy as np
import os
from PIL import Image

os.system('python3 clearsimulation.py')
logging.basicConfig(
    filename='logs/RandomAction.log',
    filemode='w',
    format='%(level)s: %(message)s',
    level=logging.INFO
)

rng = np.random.default_rng(478203)
def random_action():
    action = rng.random((4,))
    return action

env = QuadEnv('quad.xml')

FRAMERATE = 30
done = False
state = env.reset()
count = 0

while not done:
    if count < env.physics.data.time * FRAMERATE:
        im = env.render()
        im = Image.fromarray(im)
        im.save(f'simulation/frame{count:03}.jpeg')
        count += 1
    state, reward, done, info = env.step(random_action())
im = env.render()
im = Image.fromarray(im)
im.save(f'simulation/frame{count:03}.jpeg')