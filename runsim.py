from dm_control import mujoco
from PIL import Image
import os
from training import QuadEnv

FRAMERATE = 30
REPLAY_MEMORY_SIZE = 2400
MAX_EPISODES = 1000
MAX_EPISODE_LEN = 24000 # 60(sec)/0.005(timestep)*10(12,000 otherwise)
replay_memory = []

env = QuadEnv('quad.xml')

def sim_dir_cleanup():
    os.system('python3 clearsimulation.py')
    if not os.access('simulation',os.F_OK):
        os.mkdir('simulation')

def simulate():
    count = 0
    physics.reset()
    while round(physics.data.time,3) < duration:
        physics.step()
        if physics.data.ncon > 0:
            break
        flie.write(str(round(physics.data.time,3))+str(physics.data.contact)+'\n')
        if count < physics.data.time * framerate:
            print(physics.data.time*framerate)
            pixels = physics.render(height=480,width=640,camera_id='fixed_camera')
            im = Image.fromarray(pixels)
            im.save(f"simulation/frame{count:03}.jpeg")
            count += 1
    return detect_contact(physics)

def main():
    env = QuadEnv()