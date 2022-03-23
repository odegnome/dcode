from dm_control import mujoco
from PIL import Image
import os

# to create a video
# ffmpeg -framerate 30 -i 'simulation/frame%03d.jpeg' video.mpg

framerate = 30
duration = 2

physics = mujoco.Physics.from_xml_path('quad.xml')

def sim_dir_cleanup():
    os.system('python3 clearsimulation.py')
    if not os.access('simulation',os.F_OK):
        os.mkdir('simulation')

def simulate():
    count = 0
    with physics.reset_context():
        physics.set_control([0.44,0.44,0.33,0.33])

    while round(physics.data.time,3) < duration:
        physics.step()
        if count < physics.data.time * framerate:
            pixels = physics.render(height=480,width=640,camera_id='fixed_camera')
            im = Image.fromarray(pixels)
            im.save(f"simulation/frame{count:03}.jpeg")
            count += 1
        if physics.data.ncon > 0:
            detect_contact(physics)
            break
    print(count)

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
sim_dir_cleanup()
simulate()