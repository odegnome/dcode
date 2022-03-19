from dm_control import mujoco
from PIL import Image
import os
from woComposer import QuadEnv

def sim_dir_cleanup():
    os.system('python3 clearsimulation.py')
    if not os.access('simulation',os.F_OK):
        os.mkdir('simulation')

physics = mujoco.Physics.from_xml_path('quad.xml')

FRAMERATE = 30
REPLAY_MEMORY_SIZE = 2400
MAX_EPISODES = 1000
MAX_EPISODE_LEN = 24000 # 60(sec)/0.005(timestep)*10(12,000 otherwise)
replay_memory = []

def simulate():
    flie = open('logs.txt','w')
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
    flie.close()
    return detect_contact(physics)

def detect_contact(physics):
    print(physics.data.time)
    print(physics.data.ncon)
    for i_con in range(physics.data.ncon):
        id_geom1 = physics.data.contact[i_con].geom1
        id_geom2 = physics.data.contact[i_con].geom2
        name_geom1 = physics.model.id2name(id_geom1,'geom')
        name_geom2 = physics.model.id2name(id_geom2,'geom')
        print(name_geom1, name_geom2)
    return 0

def main():
    env = QuadEnv()