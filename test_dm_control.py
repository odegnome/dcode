from dm_control import mujoco
from dm_control import viewer
import numpy as np
from PIL import Image

# Load a model from an MJCF XML path.
physics = mujoco.Physics.from_xml_path('quad.xml')

physics.step()

# Print the orientation of the geoms.
# print(physics.named.data.xquat)
# print(physics.data.xquat[6])
# print(physics.named.data.xquat['quad'])
# print(physics.data.actuator_force)

a = np.concatenate((physics.data.sensordata, physics.data.xquat[6], physics.data.qvel, physics.data.actuator_force))
# print(a.shape) # (34,)

physics.set_control([0.22,0.22,0.22,0.22])
for _ in range(800):
    physics.step()

pixels = physics.render(height=480,width=640,camera_id='fixed_camera')
Image.fromarray(pixels).save('before_reset.png')

physics.reset()
pixels = physics.render(height=480,width=640,camera_id='fixed_camera')
Image.fromarray(pixels).save('after_reset.png')

# print("Sensor data", len(physics.data.sensordata))
# print("Orientaton: ", len(physics.data.xquat[6]))
# print("Lin/Ang Velocity: ", len(physics.data.qvel))
# print("Rotor control: ", len(physics.data.actuator_force))
