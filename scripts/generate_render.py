from PIL import Image
import numpy as np
# import tensorflow as tf
from Env import QuadEnv 
import os
import pickle
from dm_control.utils.transformations import quat_to_euler

# ffmpeg -framerate 30 -pattern_type glob -i 'simulation/*.jpeg' -r 30 video.mpg

seed = 89343
rng = np.random.default_rng(seed)
def check_dir_access(testrun):
    if not os.access(f'../random/{testrun}', os.F_OK):
        os.mkdir(f'../random/{testrun}')

    if not os.access(f'../random/{testrun}/frames', os.F_OK):
        os.mkdir(f'../random/{testrun}/frames')

env = QuadEnv('quad.xml')

# actor = tf.keras.models.load_model('../training/03-05/actor10000')

def get_action(state):
    # tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    # output = actor(tf_state, training=False)
    action = rng.normal(0.5, 0.1, size=4)
    action = np.clip(action, 0, 1)
    return action


# def sim_dir_cleanup():
#     os.system('python3 clearsimulation.py')

# sim_dir_cleanup()
framerate = 30

for testrun in range(10):
    done = False
    state = env.reset()
    count = 0
    debug_dict = {'rewards':[], 'actions':[], 'eulers':[]}
    check_dir_access(str(testrun))
    while not done:
        action = get_action(state)
        state, reward, done = env.step(action)
        euler_angles = quat_to_euler(env.physics.data.xquat[6].copy())
        debug_dict['eulers'].append(euler_angles.copy())
        debug_dict['actions'].append(action.copy())
        debug_dict['rewards'].append(reward)
        if count < env.physics.data.time * framerate:
            pixels = env.physics.render(height=720,width=1024,camera_id='fixed_camera')
            im = Image.fromarray(pixels)
            im.save(f"../random/{testrun}/frames/frame{count:03}.jpeg")
            count += 1
    with open(f'../random/{testrun}/debug.pickle', 'w+b') as fp:
        pickle.dump(debug_dict, fp)
    debug_dict.clear()
