from dm_control import mujoco
from PIL import Image
import os

def sim_dir_cleanup():
    os.system('python3 clearsimulation.py')
    if not os.access('simulation',os.F_OK):
        os.mkdir('simulation')

physics = mujoco.Physics.from_xml_path('quad.xml')

duration = 1
framerate = 30

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

def dqn():
    replay_memory = []
    REPLAY_MEMORY_SIZE = 5000
    MAX_EPISODES = 1000
    MAX_EPISODE_LEN = 1200 # 60(sec)/0.005(timestep)*10(12,000 otherwise)

    # Initialize Q function with random weights

    for episode in range(1,MAX_EPISODES):
        # Initialize seq s1 = {x1} and preprocessed seq phi1 = phi(s1)

        for timestep in range(1, MAX_EPISODE_LEN):

            prob = random.random()
            if prob < epsilon:
                # select a random action. Choose random value for each rotor
                pass
            else:
                # select action greedily
                pass
            
            # Take action and observe reward and next state

            # set st+1 = s_t,a_t

            # store (s_t,a_t,r_t,s_t+1) in replay_memory
            replay_memory.append(tuple(state,action,reward,next_state))

            # sample random minibatch from replay_memory

            # if next state is terminal then set y_j to reward
            # else set it to learned value
            if done: # new_state is terminal
                y_j = reward
            else:
                y_j = reward + gamma*max(Q[new_state])

            # Perform gradient descent on (y_j - Q(s_j,a_j))^2
