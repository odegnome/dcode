from dm_control import mujoco
from PIL import Image
import os

os.system('python3 clearsimulation.py')
if not os.access('simulation',os.F_OK):
    os.mkdir('simulation')

physics = mujoco.Physics.from_xml_path('quad.xml')

duration = 2
framerate = 30

count = 0
physics.reset()
while physics.data.time < duration:
    physics.step()
    if count < physics.data.time * framerate:
        print(physics.data.time*framerate)
        #pixels = physics.render(height=480,width=640,camera_id='fixed_camera')
        #im = Image.fromarray(pixels)
        #im.save(f"simulation/frame{count:03}.jpeg")
        count += 1
