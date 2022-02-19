from dm_control import mujoco
from PIL import Image

physics = mujoco.Physics.from_xml_path('quad.xml')

pixels = physics.render(height=480,width=640,camera_id='fixed_camera')

im = Image.fromarray(pixels)

im.save("pic.jpeg")
