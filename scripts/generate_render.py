from PIL import Image
import numpy as np
import tensorflow as tf
from Env import QuadEnv
import os

# ffmpeg -framerate 30 -pattern_type glob -i 'simulation/*.jpeg' video.mpg

env = QuadEnv('quad.xml')

actor = tf.keras.models.load_model('../training/temp/actor500/')

def get_action(state):
    tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    output = actor(tf_state, training=False)
    output = output.numpy()
    action = np.clip(output, 0, 1)
    return action

def sim_dir_cleanup():
    os.system('python3 clearsimulation.py')

sim_dir_cleanup()
framerate = 30

done = False
state = env.reset()
count = 0
while not done:
    action = get_action(state)
    state, reward, done, _ = env.step(action)
    if count < env.physics.data.time * framerate:
        pixels = env.physics.render(height=720,width=1024,camera_id='fixed_camera')
        im = Image.fromarray(pixels)
        im.save(f"../simulation/frame{count:03}.jpeg")
        count += 1