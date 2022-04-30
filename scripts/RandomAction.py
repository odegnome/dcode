import logging
logging.basicConfig(
    filename='../logs/RandomAction.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
import numpy as np
import os
from PIL import Image
from dm_control.utils.transformations import quat_to_euler
from Env import QuadEnv

os.system('python3 clearsimulation.py')

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
        im.save(f'../simulation/frame{count:03}.jpeg')
        count += 1
    state, reward, done, info = env.step(random_action())
    atitude_quat = env.physics.data.xquat[6].copy()
    atitude_euler = quat_to_euler(atitude_quat)
    x,y,z = atitude_euler
    logging.info(f'{atitude_euler}')
    if x > 1.4 or y > 1.4 or z > 3:
        done = True