from dm_control import mujoco
from dm_control.utils.transformations import quat_to_euler
import numpy as np
from PIL import Image

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
# orient = model.data.xquat[6].copy()
# euler = quat_to_euler(orient)
# print(np.sqrt(np.sum(np.square(euler[:2]))))

def simulate_random_initial():
    rng = np.random.default_rng(seed=73)
    framerate = 30
    duration = 1
    count = 0
    for _ in range(10):
        counter = 0
        # Reset environment
        with model.reset_context():
            model.data.qpos[:2] = rng.normal(0.0, 1.0, size=2)
            model.data.qpos[2] = rng.uniform(0.5, 4.5)
            model.data.qvel[:] = rng.normal(0.0, 1.0, size=6)

        # Simulate for 1 second
        while round(model.data.time,3) < duration:
            model.step()
            if counter < model.data.time*framerate:
                pixels=model.render(height=720, width=1024, camera_id='fixed_camera')
                im = Image.fromarray(pixels)
                im.save(f'../Frames/{count:04}.jpeg')
                count += 1
                counter += 1
    return 0

if __name__ == "__main__":
    simulate_random_initial()