from dm_control import mujoco
from dm_control.utils.transformations import quat_to_euler
import numpy as np

model = mujoco.Physics.from_xml_path('quad.xml')

# print(model.find('geom', 'quadbody').get_attributes())
# print(model.find('body', 'quad').get_attributes())
# def sensor_data():
#     sensor_array = model.data.sensordata[6:].copy()
#     sensor_array[:] = sensor_array/3
#     for index, x in enumerate(sensor_array):
#         if x<0.34:
#             sensor_array[index] = -14
#     reward = np.sum(sensor_array)
#     return reward, sensor_array

# print(model.named.data.sensordata)
# sensor_data = model.data.sensordata.copy()
# print(sensor_data)

# sensor_data[6:] = sensor_data[6:]/3
# print(sensor_data)
# sdata = list(map(lambda x: round(x, 3), sensor_data))
# print(sdata)
orient = model.data.xquat[6].copy()
euler = quat_to_euler(orient)
print(np.sqrt(np.sum(np.square(euler[:2]))))