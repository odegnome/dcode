from dm_control import mujoco
from dm_control import viewer

# Load a model from an MJCF XML path.
physics = mujoco.Physics.from_xml_path('quad.xml')

physics.step()

# Print the orientation of the geoms.
print(physics.named.data.xquat)
print(physics.data.xquat[6])
print(physics.named.data.xquat['quad'])